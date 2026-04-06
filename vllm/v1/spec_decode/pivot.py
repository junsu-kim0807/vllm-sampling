# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone pivot proposer for speculative decoding.

Pivot method semantics:
- intermediate model emits exactly one pivot token.
- draft model emits the remaining tail tokens conditioned on recovery+pivot.
- target model verifies the concatenated candidate bundle.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing_extensions import override

from vllm.config import VllmConfig, replace
from vllm.model_executor.model_loader import get_model
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID
from vllm.v1.spec_decode.adaptive_cascade import _propose_chunk_from_prefix
from vllm.v1.spec_decode.adaptive_spechive import (
    _build_hybrid_bundle_from_rows,
    _flatten_prob_rows_for_output,
)
from vllm.v1.spec_decode.draft_model import DraftModelProposer
from vllm.v1.spec_decode.spec_stage_runtime import HybridProposalBundle
from vllm.v1.spec_decode.spec_stage_utils import slice_sampling_metadata_for_subbatch
from vllm.v1.spec_decode.utils import create_vllm_config_for_draft_model

def _chain_spec_token_tree(num_tokens: int) -> str:
    return str([(i + 1) * (0,) for i in range(num_tokens)])


def _vllm_as_plain_draft(base: VllmConfig, *, length: int) -> VllmConfig:
    spec = base.speculative_config
    assert spec is not None
    new_spec = replace(
        spec,
        method="draft_model",
        num_speculative_tokens=length,
        speculative_token_tree=_chain_spec_token_tree(length),
    )
    return replace(base, speculative_config=new_spec)


def _vllm_intermediate_as_draft(base: VllmConfig, *, length: int) -> VllmConfig:
    spec = base.speculative_config
    assert spec is not None
    assert spec.intermediate_model_config is not None
    assert spec.intermediate_parallel_config is not None
    assert spec.intermediate_tensor_parallel_size is not None
    new_spec = replace(
        spec,
        method="draft_model",
        draft_model_config=spec.intermediate_model_config,
        draft_parallel_config=spec.intermediate_parallel_config,
        draft_tensor_parallel_size=spec.intermediate_tensor_parallel_size,
        num_speculative_tokens=length,
        speculative_token_tree=_chain_spec_token_tree(length),
    )
    return replace(base, speculative_config=new_spec)


class IntermediatePivotModelProposer(DraftModelProposer):
    """Loads intermediate weights under separate model tag."""

    @override
    def _get_model(self) -> nn.Module:
        from vllm.compilation.backends import set_model_tag

        temp_vllm_config = create_vllm_config_for_draft_model(self.vllm_config)
        with set_model_tag("intermediate_model"):
            return get_model(
                vllm_config=temp_vllm_config,
                prefix="intermediate_model",
            )


class PivotProposer:
    """Standalone proposer for `method="pivot"`."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner: object | None = None,
    ) -> None:
        self.vllm_config = vllm_config
        self.device = device
        self.runner = runner
        spec = vllm_config.speculative_config
        assert spec is not None
        self._L = int(spec.num_speculative_tokens)
        self._tail_len = max(0, self._L - 1)
        # L=1 is a valid pivot-only configuration (no draft tail).
        self._draft: DraftModelProposer | None = None
        if self._tail_len > 0:
            self._draft = DraftModelProposer(
                _vllm_as_plain_draft(vllm_config, length=self._tail_len),
                device,
                runner,
            )
        self._intermediate = IntermediatePivotModelProposer(
            _vllm_intermediate_as_draft(vllm_config, length=1),
            device,
            runner,
        )
        self._pending_hybrid_bundle: HybridProposalBundle | None = None

    def _main_delegate(self) -> DraftModelProposer:
        # For common runtime hooks (prepare_next_token_ids/prepare_inputs/...),
        # either proposer is valid; use draft when present, else intermediate.
        return self._draft if self._draft is not None else self._intermediate

    def __getattr__(self, name: str):
        if name.startswith("_") or name in (
            "propose",
            "load_model",
            "model",
            "intermediate_model",
            "initialize_attn_backend",
            "initialize_cudagraph_keys",
            "dummy_run",
            "validate_same_kv_cache_group",
        ):
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._main_delegate(), name)

    @property
    def model(self) -> nn.Module:
        return self._main_delegate().model

    @property
    def intermediate_model(self) -> nn.Module:
        return self._intermediate.model

    def load_model(self, target_model: nn.Module) -> None:
        if self._draft is not None:
            self._draft.load_model(target_model)
        self._intermediate.load_model(target_model)

    def initialize_attn_backend(self, kv_cache_config, kernel_block_sizes=None) -> None:
        if self._draft is not None:
            self._draft.initialize_attn_backend(kv_cache_config, kernel_block_sizes)
        self._intermediate.initialize_attn_backend(kv_cache_config, kernel_block_sizes)

    def initialize_cudagraph_keys(self, cudagraph_mode) -> None:
        if self._draft is not None:
            self._draft.initialize_cudagraph_keys(cudagraph_mode)
        self._intermediate.initialize_cudagraph_keys(cudagraph_mode)

    def dummy_run(self, *args, **kwargs):
        if self._draft is not None:
            self._draft.dummy_run(*args, **kwargs)
        self._intermediate.dummy_run(*args, **kwargs)

    def validate_same_kv_cache_group(self, kv_cache_config) -> None:
        if self._draft is not None:
            self._draft.validate_same_kv_cache_group(kv_cache_config)
        self._intermediate.validate_same_kv_cache_group(kv_cache_config)

    def clear_draft_probs(self) -> None:
        if self._draft is not None:
            self._draft.clear_draft_probs()
        self._intermediate.clear_draft_probs()

    def take_pending_spechive_bundle(self) -> HybridProposalBundle | None:
        bundle = self._pending_hybrid_bundle
        self._pending_hybrid_bundle = None
        return bundle

    def discard_pending_hierarchical_verification_state(self) -> None:
        self._pending_hybrid_bundle = None

    def _get_pivot_probs(
        self, batch_size: int, use_draft_probs: bool
    ) -> torch.Tensor | None:
        if not use_draft_probs:
            return None
        probs_flat = getattr(self._intermediate, "last_draft_probs_flat", None)
        if probs_flat is None or probs_flat.numel() == 0:
            return None
        return probs_flat.view(batch_size, -1, probs_flat.shape[-1])[:, :1].to(torch.float32)

    def propose(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
        slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        if mm_embed_inputs is not None:
            raise NotImplementedError("pivot proposer does not support multimodal draft inputs yet.")

        # Stage 1: intermediate emits the pivot token from recovery-aware next_token_ids.
        pivot_rows = self._intermediate.propose(
            target_token_ids=target_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            next_token_ids=next_token_ids,
            token_indices_to_sample=token_indices_to_sample,
            common_attn_metadata=common_attn_metadata,
            sampling_metadata=sampling_metadata,
            mm_embed_inputs=None,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            slot_mappings=slot_mappings,
        ).to(torch.int32)[:, :1]
        batch_size = int(pivot_rows.shape[0])
        use_draft_probs = (
            self.vllm_config.speculative_config is not None
            and self.vllm_config.speculative_config.use_draft_probs_in_rejection
            and not sampling_metadata.all_greedy
        )
        pivot_probs = self._get_pivot_probs(batch_size, use_draft_probs)

        # Stage 2: draft tail conditioned on recovery + pivot token.
        tail_rows = torch.empty(
            (batch_size, 0), dtype=torch.int32, device=target_token_ids.device
        )
        tail_probs: torch.Tensor | None = None
        if self._tail_len > 0:
            assert self._draft is not None
            # Build prefix rows in one host transfer (instead of per-row .item()).
            pivot_list = pivot_rows.view(batch_size).tolist()
            prefix_rows = [[int(tok)] for tok in pivot_list]
            tail_sm = slice_sampling_metadata_for_subbatch(
                sampling_metadata,
                list(range(batch_size)),
                provisional_prefix_rows=prefix_rows,
                sampled_ids_only=True,
            )
            tail_rows, tail_probs = _propose_chunk_from_prefix(
                self._draft,
                cad=common_attn_metadata,
                target_token_ids=target_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                next_token_ids=next_token_ids,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
                prefix_rows=prefix_rows,
                chunk_len=self._tail_len,
                sampling_metadata=tail_sm,
                use_draft_probs=use_draft_probs,
            )

        out = torch.full(
            (batch_size, self._L),
            PLACEHOLDER_TOKEN_ID,
            dtype=torch.int32,
            device=target_token_ids.device,
        )
        out[:, :1] = pivot_rows
        if self._tail_len > 0:
            out[:, 1 : 1 + self._tail_len] = tail_rows

        source_stage_rows = [[0, *([1] * self._tail_len)] for _ in range(batch_size)]

        draft_probs_flat: torch.Tensor | None = None
        if use_draft_probs:
            if pivot_probs is not None and (self._tail_len == 0 or tail_probs is not None):
                prob_rows: list[list[torch.Tensor]] = []
                for b in range(batch_size):
                    row = [pivot_probs[b, 0]]
                    if self._tail_len > 0 and tail_probs is not None:
                        row.extend([tail_probs[b, j] for j in range(self._tail_len)])
                    prob_rows.append(row)
                draft_probs_flat = _flatten_prob_rows_for_output(prob_rows, out)

        self._pending_hybrid_bundle = _build_hybrid_bundle_from_rows(
            out,
            mode="pivot",
            draft_probs=draft_probs_flat,
            source_stage_rows=source_stage_rows,
        )
        self.clear_draft_probs()
        return out
