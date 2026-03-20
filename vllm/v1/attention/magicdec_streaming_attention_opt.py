# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optimized MagicDec streaming sparse KV view for draft attention."""

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

            self._cols = torch.arange(self._keep_block_cap, device=device, dtype=torch.long)
            self._recent_cols = (
                torch.arange(self._num_recent_blocks, device=device, dtype=torch.long)
                if self._num_recent_blocks > 0
                else None
            )

            self._sparse_block_tables = torch.zeros(
                (scheduler_config.max_num_seqs, self._keep_block_cap),
                device=device,
                dtype=torch.int32,
            )
            self._sparse_seq_lens = torch.zeros(
                (scheduler_config.max_num_seqs,), device=device, dtype=torch.int32
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

            orig_bt = common_attn_metadata.block_table_tensor[:num_reqs]
            orig_sl = common_attn_metadata.seq_lens[:num_reqs]
            bs = self._block_size
            keep = self._keep_block_cap

            num_total_blocks = torch.div(orig_sl + (bs - 1), bs, rounding_mode="floor")
            full_keep = num_total_blocks <= keep
            valid_counts = torch.clamp(num_total_blocks, max=keep)

            idx = self._logical_block_idx[:num_reqs]
            idx.copy_(self._cols.unsqueeze(0).expand(num_reqs, -1))
            if self._num_recent_blocks > 0:
                tail_idx = (
                    num_total_blocks.to(torch.long).unsqueeze(1)
                    - self._num_recent_blocks
                    + self._recent_cols.unsqueeze(0)
                )
                idx[:, self._num_sink_blocks :] = torch.where(
                    full_keep.unsqueeze(1), idx[:, self._num_sink_blocks :], tail_idx
                )

            valid = self._valid_mask[:num_reqs]
            valid.copy_(self._cols.unsqueeze(0) < valid_counts.unsqueeze(1))
            idx.masked_fill_(~valid, 0)

            out_bt = self._sparse_block_tables[:num_reqs]
            out_bt.copy_(orig_bt.gather(1, idx))
            out_bt.masked_fill_(~valid, 0)

            out_sl = self._sparse_seq_lens[:num_reqs]
            if self._num_recent_blocks > 0:
                recent_visible_tokens = torch.clamp(
                    orig_sl - (num_total_blocks - self._num_recent_blocks) * bs, min=0
                )
                truncated_visible = self._num_sink_blocks * bs + recent_visible_tokens
            else:
                truncated_visible = torch.full_like(orig_sl, self._num_sink_blocks * bs)
            out_sl.copy_(torch.where(full_keep, orig_sl, truncated_visible))

            rewritten_max_seq_len = min(
                common_attn_metadata.max_seq_len, self._max_visible_tokens
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
