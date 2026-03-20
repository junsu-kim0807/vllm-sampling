# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MagicDec streaming (StreamingLLM-style) sparse KV *view* for draft attention.

Physical KV allocation and block manager are unchanged. At attention metadata
build time we rewrite ``block_table_tensor`` and ``seq_lens`` so attention only
sees sink blocks + a recent tail of blocks.

This optimized version avoids:
1. GPU -> CPU sync from ``detach().cpu()``
2. GPU -> CPU sync from ``.item()``
3. Python per-request/per-block loops
4. Full block-table clone on every draft step
"""

import functools

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    CommonAttentionMetadata,
    subclass_attention_backend,
)

logger = init_logger(__name__)


@functools.lru_cache
def create_magicdec_streaming_attention_backend(
    underlying_attn_backend: type[AttentionBackend],
    kv_budget_tokens: int,
    block_size: int,
) -> type[AttentionBackend]:
    """Wrap `underlying_attn_backend` with a builder that rewrites metadata."""
    if kv_budget_tokens < block_size:
        raise ValueError(
            f"MagicDec streaming requires kv_budget_tokens >= block_size, "
            f"but got kv_budget_tokens={kv_budget_tokens}, block_size={block_size}."
        )

    prefix = "MagicDecStream_"
    underlying_builder = underlying_attn_backend.get_builder_cls()

    sink_tokens = block_size
    recent_tokens = kv_budget_tokens - sink_tokens
    recent_tokens = (max(0, recent_tokens) // block_size) * block_size

    num_sink_blocks_init = sink_tokens // block_size
    num_recent_blocks_init = recent_tokens // block_size if recent_tokens > 0 else 0
    keep_block_cap = num_sink_blocks_init + num_recent_blocks_init
    max_visible_tokens = keep_block_cap * block_size

    class MagicDecStreamingAttentionBuilder(underlying_builder):  # type: ignore
        # Must stay False because this wrapper changes seq_lens/max_seq_len
        # together with block_table_tensor.
        supports_update_block_table = False

        def __init__(
            self,
            kv_cache_spec,
            layer_names: list[str],
            vllm_config: VllmConfig,
            device: torch.device,
        ):
            super().__init__(kv_cache_spec, layer_names, vllm_config, device)

            scheduler_config = vllm_config.scheduler_config

            self._block_size = block_size
            self._num_sink_blocks = num_sink_blocks_init
            self._num_recent_blocks = num_recent_blocks_init
            self._keep_block_cap = keep_block_cap
            self._max_visible_tokens = max_visible_tokens

            # Small constant index tensors reused every build.
            self._cols = torch.arange(
                self._keep_block_cap,
                device=device,
                dtype=torch.long,
            )
            if self._num_recent_blocks > 0:
                self._recent_cols = torch.arange(
                    self._num_recent_blocks,
                    device=device,
                    dtype=torch.long,
                )
            else:
                self._recent_cols = None

            # Reusable scratch buffers.
            self._sparse_block_tables = torch.zeros(
                (scheduler_config.max_num_seqs, self._keep_block_cap),
                device=device,
                dtype=torch.int32,
            )
            self._sparse_seq_lens = torch.zeros(
                (scheduler_config.max_num_seqs,),
                device=device,
                dtype=torch.int32,
            )
            self._logical_block_idx = torch.zeros(
                (scheduler_config.max_num_seqs, self._keep_block_cap),
                device=device,
                dtype=torch.long,
            )
            self._valid_mask = torch.zeros(
                (scheduler_config.max_num_seqs, self._keep_block_cap),
                device=device,
                dtype=torch.bool,
            )

        def _rewrite_common_metadata(
            self, common_attn_metadata: CommonAttentionMetadata
        ) -> CommonAttentionMetadata:
            num_reqs = common_attn_metadata.num_reqs
            if num_reqs <= 0:
                return common_attn_metadata

            # Slice only active requests.
            orig_bt = common_attn_metadata.block_table_tensor[:num_reqs]
            orig_sl = common_attn_metadata.seq_lens[:num_reqs]

            bs = self._block_size
            keep = self._keep_block_cap

            # Number of logical blocks per request: ceil(seq_len / block_size)
            num_total_blocks = torch.div(
                orig_sl + (bs - 1),
                bs,
                rounding_mode="floor",
            )

            # Requests that already fit within the visible budget keep full view.
            full_keep = num_total_blocks <= keep

            # How many logical block indices are valid in the rewritten view.
            valid_counts = torch.clamp(num_total_blocks, max=keep)

            # Build logical block indices entirely on GPU.
            idx = self._logical_block_idx[:num_reqs]
            idx.copy_(self._cols.unsqueeze(0).expand(num_reqs, -1))

            if self._num_recent_blocks > 0:
                tail_idx = (
                    num_total_blocks.to(torch.long).unsqueeze(1)
                    - self._num_recent_blocks
                    + self._recent_cols.unsqueeze(0)
                )
                idx[:, self._num_sink_blocks :] = torch.where(
                    full_keep.unsqueeze(1),
                    idx[:, self._num_sink_blocks :],
                    tail_idx,
                )

            valid = self._valid_mask[:num_reqs]
            valid.copy_(self._cols.unsqueeze(0) < valid_counts.unsqueeze(1))

            # Safe placeholder for invalid gather positions.
            idx.masked_fill_(~valid, 0)

            # Gather only the needed logical blocks.
            out_bt = self._sparse_block_tables[:num_reqs]
            out_bt.copy_(orig_bt.gather(1, idx))
            out_bt.masked_fill_(~valid, 0)

            # Visible seq len after sparse view rewrite.
            out_sl = self._sparse_seq_lens[:num_reqs]

            if self._num_recent_blocks > 0:
                recent_visible_tokens = torch.clamp(
                    orig_sl - (num_total_blocks - self._num_recent_blocks) * bs,
                    min=0,
                )
                truncated_visible = self._num_sink_blocks * bs + recent_visible_tokens
            else:
                truncated_visible = torch.full_like(
                    orig_sl,
                    self._num_sink_blocks * bs,
                )

            out_sl.copy_(torch.where(full_keep, orig_sl, truncated_visible))

            # Avoid host sync from out_sl.max().item().
            rewritten_max_seq_len = min(
                common_attn_metadata.max_seq_len,
                self._max_visible_tokens,
            )

            return common_attn_metadata.replace(
                block_table_tensor=out_bt,
                seq_lens=out_sl,
                max_seq_len=rewritten_max_seq_len,
                _seq_lens_cpu=None,
                _num_computed_tokens_cpu=None,
                _num_computed_tokens_cache=None,
            )

        def build_for_drafting(
            self,
            common_attn_metadata: CommonAttentionMetadata,
            draft_index: int,
        ) -> AttentionMetadata:
            rewritten = self._rewrite_common_metadata(common_attn_metadata)
            return super().build_for_drafting(rewritten, draft_index)

    logger.info(
        "MagicDec streaming attention enabled: kv_budget_tokens=%d block_size=%d "
        "sink_blocks=%d recent_blocks=%d effective_visible_tokens=%d",
        kv_budget_tokens,
        block_size,
        num_sink_blocks_init,
        num_recent_blocks_init,
        max_visible_tokens,
    )

    return subclass_attention_backend(
        name_prefix=prefix,
        attention_backend_cls=underlying_attn_backend,
        builder_cls=MagicDecStreamingAttentionBuilder,
    )
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MagicDec streaming (StreamingLLM-style) sparse KV *view* for draft attention.

Physical KV allocation and block manager are unchanged. At attention metadata
build time we rewrite ``block_table_tensor`` and ``seq_lens`` so attention only
sees sink blocks + a recent tail of blocks.

This optimized version avoids:
1. GPU -> CPU sync from ``detach().cpu()``
2. GPU -> CPU sync from ``.item()``
3. Python per-request/per-block loops
4. Full block-table clone on every draft step
"""

import functools

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    CommonAttentionMetadata,
    subclass_attention_backend,
)

logger = init_logger(__name__)


@functools.lru_cache
def create_magicdec_streaming_attention_backend(
    underlying_attn_backend: type[AttentionBackend],
    kv_budget_tokens: int,
    block_size: int,
) -> type[AttentionBackend]:
    """Wrap `underlying_attn_backend` with a builder that rewrites metadata."""
    if kv_budget_tokens < block_size:
        raise ValueError(
            f"MagicDec streaming requires kv_budget_tokens >= block_size, "
            f"but got kv_budget_tokens={kv_budget_tokens}, block_size={block_size}."
        )

    prefix = "MagicDecStream_"
    underlying_builder = underlying_attn_backend.get_builder_cls()

    sink_tokens = block_size
    recent_tokens = kv_budget_tokens - sink_tokens
    recent_tokens = (max(0, recent_tokens) // block_size) * block_size

    num_sink_blocks_init = sink_tokens // block_size
    num_recent_blocks_init = recent_tokens // block_size if recent_tokens > 0 else 0
    keep_block_cap = num_sink_blocks_init + num_recent_blocks_init
    max_visible_tokens = keep_block_cap * block_size

    class MagicDecStreamingAttentionBuilder(underlying_builder):  # type: ignore
        # Must stay False because this wrapper changes seq_lens/max_seq_len
        # together with block_table_tensor.
        supports_update_block_table = False

        def __init__(
            self,
            kv_cache_spec,
            layer_names: list[str],
            vllm_config: VllmConfig,
            device: torch.device,
        ):
            super().__init__(kv_cache_spec, layer_names, vllm_config, device)

            scheduler_config = vllm_config.scheduler_config

            self._kv_budget_tokens = kv_budget_tokens
            self._block_size = block_size
            self._num_sink_blocks = num_sink_blocks_init
            self._num_recent_blocks = num_recent_blocks_init
            self._keep_block_cap = keep_block_cap
            self._max_visible_tokens = max_visible_tokens

            # Small constant index tensors reused every build.
            self._cols = torch.arange(
                self._keep_block_cap,
                device=device,
                dtype=torch.long,
            )
            if self._num_recent_blocks > 0:
                self._recent_cols = torch.arange(
                    self._num_recent_blocks,
                    device=device,
                    dtype=torch.long,
                )
            else:
                self._recent_cols = None

            # Reusable scratch buffers.
            self._sparse_block_tables = torch.zeros(
                (scheduler_config.max_num_seqs, self._keep_block_cap),
                device=device,
                dtype=torch.int32,
            )
            self._sparse_seq_lens = torch.zeros(
                (scheduler_config.max_num_seqs,),
                device=device,
                dtype=torch.int32,
            )
            self._logical_block_idx = torch.zeros(
                (scheduler_config.max_num_seqs, self._keep_block_cap),
                device=device,
                dtype=torch.long,
            )
            self._valid_mask = torch.zeros(
                (scheduler_config.max_num_seqs, self._keep_block_cap),
                device=device,
                dtype=torch.bool,
            )

        def _rewrite_common_metadata(
            self, common_attn_metadata: CommonAttentionMetadata
        ) -> CommonAttentionMetadata:
            num_reqs = common_attn_metadata.num_reqs
            if num_reqs <= 0:
                return common_attn_metadata

            # Slice only active requests.
            orig_bt = common_attn_metadata.block_table_tensor[:num_reqs]
            orig_sl = common_attn_metadata.seq_lens[:num_reqs]

            bs = self._block_size
            keep = self._keep_block_cap

            # Number of logical blocks per request: ceil(seq_len / block_size)
            num_total_blocks = torch.div(
                orig_sl + (bs - 1),
                bs,
                rounding_mode="floor",
            )

            # Requests that already fit within the visible budget keep full view.
            full_keep = num_total_blocks <= keep

            # How many logical block indices are valid in the rewritten view.
            valid_counts = torch.clamp(num_total_blocks, max=keep)

            # Build logical block indices entirely on GPU.
            #
            # full_keep row:
            #   [0, 1, 2, ..., num_total_blocks-1]
            #
            # truncated row:
            #   [sink blocks] + [recent tail blocks]
            idx = self._logical_block_idx[:num_reqs]
            idx.copy_(self._cols.unsqueeze(0).expand(num_reqs, -1))

            if self._num_recent_blocks > 0:
                tail_idx = (
                    num_total_blocks.to(torch.long).unsqueeze(1)
                    - self._num_recent_blocks
                    + self._recent_cols.unsqueeze(0)
                )
                idx[:, self._num_sink_blocks :] = torch.where(
                    full_keep.unsqueeze(1),
                    idx[:, self._num_sink_blocks :],
                    tail_idx,
                )

            valid = self._valid_mask[:num_reqs]
            valid.copy_(self._cols.unsqueeze(0) < valid_counts.unsqueeze(1))

            # Safe placeholder for invalid gather positions.
            idx.masked_fill_(~valid, 0)

            # Gather only the needed logical blocks.
            out_bt = self._sparse_block_tables[:num_reqs]
            out_bt.copy_(orig_bt.gather(1, idx))
            out_bt.masked_fill_(~valid, 0)

            # Visible seq len after sparse view rewrite.
            #
            # full_keep:
            #   unchanged
            #
            # truncated:
            #   sink_full_tokens + visible_recent_tokens
            out_sl = self._sparse_seq_lens[:num_reqs]

            if self._num_recent_blocks > 0:
                recent_visible_tokens = torch.clamp(
                    orig_sl - (num_total_blocks - self._num_recent_blocks) * bs,
                    min=0,
                )
                truncated_visible = self._num_sink_blocks * bs + recent_visible_tokens
            else:
                truncated_visible = torch.full_like(
                    orig_sl,
                    self._num_sink_blocks * bs,
                )

            out_sl.copy_(torch.where(full_keep, orig_sl, truncated_visible))

            # Avoid host sync from out_sl.max().item().
            # Safe upper bound:
            #   out_sl <= original max_seq_len
            #   out_sl <= max_visible_tokens
            rewritten_max_seq_len = min(
                common_attn_metadata.max_seq_len,
                self._max_visible_tokens,
            )

            return common_attn_metadata.replace(
                block_table_tensor=out_bt,
                seq_lens=out_sl,
                max_seq_len=rewritten_max_seq_len,
                _seq_lens_cpu=None,
                _num_computed_tokens_cpu=None,
                _num_computed_tokens_cache=None,
            )

        def build_for_drafting(
            self,
            common_attn_metadata: CommonAttentionMetadata,
            draft_index: int,
        ) -> AttentionMetadata:
            rewritten = self._rewrite_common_metadata(common_attn_metadata)
            return super().build_for_drafting(rewritten, draft_index)

    logger.info(
        "MagicDec streaming attention enabled: kv_budget_tokens=%d block_size=%d "
        "sink_blocks=%d recent_blocks=%d effective_visible_tokens=%d",
        kv_budget_tokens,
        block_size,
        num_sink_blocks_init,
        num_recent_blocks_init,
        max_visible_tokens,
    )

    return subclass_attention_backend(
        name_prefix=prefix,
        attention_backend_cls=underlying_attn_backend,
        builder_cls=MagicDecStreamingAttentionBuilder,
    )
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MagicDec streaming (StreamingLLM-style) sparse KV *view* for draft attention.

Physical KV allocation and block manager are unchanged. At attention metadata
build time we rewrite ``block_table_tensor`` and ``seq_lens`` (and
``max_seq_len``) so attention only sees sink blocks + a recent tail of blocks,
dropping middle blocks — same pattern as StaticSinkAttention (view rewrite in
the builder before delegating to the underlying backend).
"""

import functools

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    CommonAttentionMetadata,
    subclass_attention_backend,
)

logger = init_logger(__name__)


def _streaming_block_indices(
    *,
    seq_len: int,
    block_size: int,
    num_sink_blocks: int,
    num_recent_blocks: int,
) -> list[int]:
    """Return ordered unique logical block indices to keep (sink + recent)."""
    if seq_len <= 0:
        return []
    num_total = cdiv(seq_len, block_size)
    if num_total <= num_sink_blocks + num_recent_blocks:
        return list(range(num_total))

    sink_idx = list(range(min(num_sink_blocks, num_total)))
    start_recent = max(num_sink_blocks, num_total - num_recent_blocks)
    recent_idx = list(range(start_recent, num_total))

    ordered: list[int] = []
    seen: set[int] = set()
    for i in sink_idx + recent_idx:
        if 0 <= i < num_total and i not in seen:
            ordered.append(i)
            seen.add(i)
    return ordered


def _visible_seq_len(
    selected_blocks: list[int], seq_len: int, block_size: int
) -> int:
    total = 0
    for b in selected_blocks:
        block_start = b * block_size
        block_end = min(seq_len, (b + 1) * block_size)
        total += block_end - block_start
    return total


@functools.lru_cache
def create_magicdec_streaming_attention_backend(
    underlying_attn_backend: type[AttentionBackend],
    kv_budget_tokens: int,
    block_size: int,
) -> type[AttentionBackend]:
    """Wrap `underlying_attn_backend` with a builder that rewrites metadata."""
    if kv_budget_tokens < block_size:
        raise ValueError(
            f"MagicDec streaming requires kv_budget_tokens >= block_size, "
            f"but got kv_budget_tokens={kv_budget_tokens}, block_size={block_size}."
        )

    prefix = "MagicDecStream_"
    underlying_builder = underlying_attn_backend.get_builder_cls()
    sink_tokens = block_size
    recent_tokens = kv_budget_tokens - sink_tokens
    recent_tokens = (max(0, recent_tokens) // block_size) * block_size
    num_sink_blocks_init = sink_tokens // block_size
    num_recent_blocks_init = (
        recent_tokens // block_size if recent_tokens > 0 else 0
    )

    class MagicDecStreamingAttentionBuilder(underlying_builder):  # type: ignore
        # Must be False because this wrapper changes seq_lens/max_seq_len in
        # addition to block_table_tensor. Underlying update_block_table() paths
        # only update block_table/slot_mapping and would leave stale lengths.
        supports_update_block_table = False

        def __init__(
            self,
            kv_cache_spec,
            layer_names: list[str],
            vllm_config: VllmConfig,
            device: torch.device,
        ):
            super().__init__(kv_cache_spec, layer_names, vllm_config, device)
            self._kv_budget_tokens = kv_budget_tokens
            self._block_size = block_size
            self._num_sink_blocks = num_sink_blocks_init
            self._num_recent_blocks = num_recent_blocks_init

        def _rewrite_common_metadata(
            self, common_attn_metadata: CommonAttentionMetadata
        ) -> CommonAttentionMetadata:
            num_reqs = common_attn_metadata.num_reqs
            if num_reqs <= 0:
                return common_attn_metadata

            orig_bt = common_attn_metadata.block_table_tensor
            orig_sl = common_attn_metadata.seq_lens
            out_bt = orig_bt.clone()
            out_sl = orig_sl.clone()

            seq_lens_cpu = orig_sl[:num_reqs].detach().cpu()
            block_size = self._block_size

            for r in range(num_reqs):
                seq_len = int(seq_lens_cpu[r].item())
                if seq_len <= 0:
                    out_bt[r].zero_()
                    out_sl[r] = 0
                    continue

                selected = _streaming_block_indices(
                    seq_len=seq_len,
                    block_size=block_size,
                    num_sink_blocks=self._num_sink_blocks,
                    num_recent_blocks=self._num_recent_blocks,
                )
                vis = _visible_seq_len(selected, seq_len, block_size)
                row = orig_bt[r]
                out_bt[r].zero_()
                for j, bi in enumerate(selected):
                    out_bt[r, j] = row[bi]
                out_sl[r] = vis

            batch_max = int(out_sl[:num_reqs].max().item())
            if batch_max <= 0:
                batch_max = int(common_attn_metadata.max_seq_len)

            return common_attn_metadata.replace(
                block_table_tensor=out_bt,
                seq_lens=out_sl,
                max_seq_len=batch_max,
                _seq_lens_cpu=None,
                _num_computed_tokens_cpu=None,
                _num_computed_tokens_cache=None,
            )

        def build_for_drafting(
            self,
            common_attn_metadata: CommonAttentionMetadata,
            draft_index: int,
        ) -> AttentionMetadata:
            # Apply MagicDec sparse KV view only on the drafting path.
            # Do not rewrite the generic build() path, because prompt prefill
            # and non-drafting extend paths may require full causal visibility.
            rewritten = self._rewrite_common_metadata(common_attn_metadata)
            return super().build_for_drafting(rewritten, draft_index)

    logger.info(
        "MagicDec streaming attention enabled: kv_budget_tokens=%d block_size=%d "
        "sink_blocks=%d recent_blocks=%d effective_visible_tokens=%d",
        kv_budget_tokens,
        block_size,
        num_sink_blocks_init,
        num_recent_blocks_init,
        (num_sink_blocks_init + num_recent_blocks_init) * block_size,
    )

    return subclass_attention_backend(
        name_prefix=prefix,
        attention_backend_cls=underlying_attn_backend,
        builder_cls=MagicDecStreamingAttentionBuilder,
    )
