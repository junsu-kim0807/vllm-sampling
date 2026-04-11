# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Staged Eagle3 helpers for pivot runtime.

This module keeps Eagle3-specific hidden-state handling localized:
- Intermediate provider owns provisional staged hidden states.
- Eagle head proposal engine consumes raw bundles and combines internally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from vllm.forward_context import set_forward_context
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.adaptive_cascade import (
    _build_prefix_conditioned_inputs,
    _hv_clone_cad,
    _propose_chunk_from_prefix,
)
from vllm.v1.spec_decode.spec_stage_runtime import (
    DitRoundProposal,
    IntermediateRoundState,
    StagedHiddenStateBundle,
)


class ProposalEngine(Protocol):
    def propose_chunk_from_prefix(
        self,
        *,
        base_target_token_ids: torch.Tensor,
        base_target_positions: torch.Tensor,
        base_target_hidden_states: torch.Tensor,
        base_next_token_ids: torch.Tensor,
        base_common_attn_metadata: CommonAttentionMetadata,
        base_num_rejected_tokens_gpu: torch.Tensor | None,
        prefix_rows: list[list[int]],
        chunk_len: int,
        sampling_metadata: SamplingMetadata,
        use_draft_probs: bool,
    ) -> DitRoundProposal: ...


class HiddenStateProvider(Protocol):
    def bootstrap_from_current_prefix(
        self,
        *,
        base_target_token_ids: torch.Tensor,
        base_target_positions: torch.Tensor,
        base_target_hidden_states: torch.Tensor,
        base_next_token_ids: torch.Tensor,
        base_common_attn_metadata: CommonAttentionMetadata,
        base_num_rejected_tokens_gpu: torch.Tensor | None,
        prefix_rows: list[list[int]],
    ) -> IntermediateRoundState: ...


@dataclass
class DraftModelProposalEngine:
    proposer: object

    def propose_chunk_from_prefix(
        self,
        *,
        base_target_token_ids: torch.Tensor,
        base_target_positions: torch.Tensor,
        base_target_hidden_states: torch.Tensor,
        base_next_token_ids: torch.Tensor,
        base_common_attn_metadata: CommonAttentionMetadata,
        base_num_rejected_tokens_gpu: torch.Tensor | None,
        prefix_rows: list[list[int]],
        chunk_len: int,
        sampling_metadata: SamplingMetadata,
        use_draft_probs: bool,
    ) -> DitRoundProposal:
        rows, probs = _propose_chunk_from_prefix(
            self.proposer,  # type: ignore[arg-type]
            cad=base_common_attn_metadata,
            target_token_ids=base_target_token_ids,
            target_positions=base_target_positions,
            target_hidden_states=base_target_hidden_states,
            next_token_ids=base_next_token_ids,
            num_rejected_tokens_gpu=base_num_rejected_tokens_gpu,
            prefix_rows=prefix_rows,
            chunk_len=chunk_len,
            sampling_metadata=sampling_metadata,
            use_draft_probs=use_draft_probs,
        )
        return DitRoundProposal(tokens=rows, probs=probs)


@dataclass
class EagleHeadProposalEngine:
    proposer: object

    def propose_from_hidden_states(
        self,
        *,
        base_target_token_ids: torch.Tensor,
        base_target_positions: torch.Tensor,
        hidden_bundle: StagedHiddenStateBundle,
        base_next_token_ids: torch.Tensor,
        base_common_attn_metadata: CommonAttentionMetadata,
        base_num_rejected_tokens_gpu: torch.Tensor | None,
        prefix_rows: list[list[int]],
        chunk_len: int,
        sampling_metadata: SamplingMetadata,
        use_draft_probs: bool,
    ) -> DitRoundProposal:
        # Eagle3-specific hidden-state combine remains in Eagle proposer path.
        rows, probs = _propose_chunk_from_prefix(
            self.proposer,  # type: ignore[arg-type]
            cad=base_common_attn_metadata,
            target_token_ids=base_target_token_ids,
            target_positions=base_target_positions,
            target_hidden_states=hidden_bundle.hidden_states,
            next_token_ids=base_next_token_ids,
            num_rejected_tokens_gpu=base_num_rejected_tokens_gpu,
            prefix_rows=prefix_rows,
            chunk_len=chunk_len,
            sampling_metadata=sampling_metadata,
            use_draft_probs=use_draft_probs,
            prefix_prefab=hidden_bundle.prefix_prefab,
        )
        return DitRoundProposal(tokens=rows, probs=probs)


@dataclass
class IntermediateModelStateProvider:
    proposer: object

    def _forward_hidden_states(
        self,
        *,
        base_target_token_ids: torch.Tensor,
        base_target_positions: torch.Tensor,
        base_target_hidden_states: torch.Tensor,
        base_next_token_ids: torch.Tensor,
        base_common_attn_metadata: CommonAttentionMetadata,
        base_num_rejected_tokens_gpu: torch.Tensor | None,
        prefix_rows: list[list[int]],
    ) -> StagedHiddenStateBundle:
        (
            pref_toks,
            pref_pos,
            pref_hidden,
            pref_next,
            pref_cad,
            _,
            _,
            _,
        ) = _build_prefix_conditioned_inputs(
            self.proposer,  # type: ignore[arg-type]
            cad=_hv_clone_cad(base_common_attn_metadata),
            target_token_ids=base_target_token_ids,
            target_positions=base_target_positions,
            target_hidden_states=base_target_hidden_states,
            next_token_ids=base_next_token_ids,
            prefix_rows=prefix_rows,
            roll_rows=None,
        )
        num_tokens, _, pref_cad = self.proposer.set_inputs_first_pass(  # type: ignore[attr-defined]
            target_token_ids=pref_toks,
            next_token_ids=pref_next,
            target_positions=pref_pos,
            target_hidden_states=pref_hidden,
            token_indices_to_sample=None,
            cad=pref_cad,
            num_rejected_tokens_gpu=base_num_rejected_tokens_gpu,
        )
        per_layer_attn_metadata: dict[str, object] = {}
        attn_metadata = None
        for attn_group in self.proposer.draft_attn_groups:  # type: ignore[attr-defined]
            attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
                common_attn_metadata=pref_cad,
                draft_index=0,
            )
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata

        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            self.proposer._determine_batch_execution_and_padding(num_tokens)  # type: ignore[attr-defined]
        )
        model_kwargs = {
            "input_ids": self.proposer.input_ids[:num_input_tokens],  # type: ignore[attr-defined]
            "positions": self.proposer._get_positions(num_input_tokens),  # type: ignore[attr-defined]
            "inputs_embeds": None,
        }
        if self.proposer.pass_hidden_states_to_model:  # type: ignore[attr-defined]
            model_kwargs["hidden_states"] = self.proposer.hidden_states[:num_input_tokens]  # type: ignore[attr-defined]
        with set_forward_context(
            per_layer_attn_metadata,
            self.proposer.vllm_config,  # type: ignore[attr-defined]
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=self.proposer._get_slot_mapping(  # type: ignore[attr-defined]
                num_input_tokens, pref_cad.slot_mapping
            ),
        ):
            ret_hidden_states = self.proposer.model(**model_kwargs)  # type: ignore[attr-defined]
            if self.proposer.model_returns_tuple():  # type: ignore[attr-defined]
                hidden_states, _ = ret_hidden_states
            else:
                hidden_states = ret_hidden_states
        out_h = hidden_states.to(torch.float32)
        return StagedHiddenStateBundle(
            hidden_states=out_h,
            aux_hidden_states=None,
            batch_size=int(pref_cad.batch_size()),
            owns_provisional_frontier=True,
            prefix_prefab=(pref_toks, pref_pos, out_h, pref_next, pref_cad),
        )

    def bootstrap_from_current_prefix(
        self,
        *,
        base_target_token_ids: torch.Tensor,
        base_target_positions: torch.Tensor,
        base_target_hidden_states: torch.Tensor,
        base_next_token_ids: torch.Tensor,
        base_common_attn_metadata: CommonAttentionMetadata,
        base_num_rejected_tokens_gpu: torch.Tensor | None,
        prefix_rows: list[list[int]],
    ) -> IntermediateRoundState:
        bundle = self._forward_hidden_states(
            base_target_token_ids=base_target_token_ids,
            base_target_positions=base_target_positions,
            base_target_hidden_states=base_target_hidden_states,
            base_next_token_ids=base_next_token_ids,
            base_common_attn_metadata=base_common_attn_metadata,
            base_num_rejected_tokens_gpu=base_num_rejected_tokens_gpu,
            prefix_rows=prefix_rows,
        )
        return IntermediateRoundState(
            hidden_bundle=bundle,
            verification_metadata={"bootstrap_source": "intermediate_model"},
            frontier_metadata=[{"row": i, "provisional": True} for i in range(bundle.batch_size)],
            bootstrap_complete=True,
        )
