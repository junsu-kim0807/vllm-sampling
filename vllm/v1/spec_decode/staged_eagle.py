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
        prefix_rows: list[list[int]] | None = None,
        prefix_tokens: torch.Tensor | None = None,
        prefix_lengths: torch.Tensor | None = None,
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
        prefix_rows: list[list[int]] | None = None,
        prefix_tokens: torch.Tensor | None = None,
        prefix_lengths: torch.Tensor | None = None,
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
            prefix_tokens=prefix_tokens,
            prefix_lengths=prefix_lengths,
        )
        # When the proposer has needs_extra_input_slots (e.g. draft_model with
        # pass_hidden_states_to_model=False), set_inputs_first_pass will call
        # extend_all_queries_by_N which shifts query_start_loc, seq_lens, and
        # num_actual_tokens by +N per row.  The prefab tensors (pref_toks,
        # pref_pos, pref_next) are NOT extended, so the prefab must store the
        # PRE-set_inputs_first_pass CAD to keep metadata consistent with the
        # stored tensors.  We also remap the model-output hidden states back
        # to the pre-sifp layout so downstream _build_prefix_conditioned_inputs
        # can index them correctly.
        has_extra_slots: bool = getattr(
            self.proposer, "needs_extra_input_slots", False
        )
        pre_sifp_cad = _hv_clone_cad(pref_cad) if has_extra_slots else None

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

        self.proposer._log_draft_attn_metadata_debug(  # type: ignore[attr-defined]
            per_layer_attn_metadata,
            "staged_eagle._forward_hidden_states:before_set_forward_context",
        )
        self.proposer._check_per_layer_attn_metadata_contract(  # type: ignore[attr-defined]
            per_layer_attn_metadata
        )

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
            # Same convention as eagle.py propose(): tuple is
            # (last_hidden_states, hidden_states); bundle must carry the second
            # for downstream target_hidden_states, not the last-layer logits view.
            if not self.proposer.model_returns_tuple():  # type: ignore[attr-defined]
                hidden_states = ret_hidden_states
            else:
                _, hidden_states = ret_hidden_states
        out_h = hidden_states.to(torch.float32)

        if pre_sifp_cad is not None:
            # The model output (out_h) is laid out according to the post-sifp
            # qsl (which has extra slots interleaved per row).  Remap to
            # pre-sifp layout so that token positions match pref_toks/pref_pos.
            pre_qsl = pre_sifp_cad.query_start_loc
            post_qsl = pref_cad.query_start_loc
            pre_num_tokens = pre_sifp_cad.num_actual_tokens
            batch_size = int(pre_sifp_cad.batch_size())
            store_h = torch.empty(
                pre_num_tokens,
                out_h.shape[-1],
                dtype=out_h.dtype,
                device=out_h.device,
            )
            for b in range(batch_size):
                orig_len = int(pre_qsl[b + 1].item()) - int(pre_qsl[b].item())
                src_start = int(post_qsl[b].item())
                dst_start = int(pre_qsl[b].item())
                store_h[dst_start : dst_start + orig_len] = out_h[
                    src_start : src_start + orig_len
                ]
            prefab_cad = pre_sifp_cad
            prefab_h = store_h
        else:
            prefab_cad = pref_cad
            prefab_h = out_h

        if pre_sifp_cad is not None:
            # Remapped prefab: flat tensors and CAD must agree on token count
            # (post-SIFP out_h may be padded; prefab_h is exact).
            _ntok = int(prefab_h.shape[0])
            assert int(prefab_cad.query_start_loc[-1].item()) == _ntok
            assert int(pref_toks.shape[0]) == _ntok
            _pos_ntok = (
                int(pref_pos.shape[0])
                if pref_pos.dim() == 1
                else int(pref_pos.shape[1])
            )
            assert _pos_ntok == _ntok

        return StagedHiddenStateBundle(
            hidden_states=prefab_h,
            aux_hidden_states=None,
            batch_size=int(prefab_cad.batch_size()),
            owns_provisional_frontier=True,
            prefix_prefab=(pref_toks, pref_pos, prefab_h, pref_next, prefab_cad),
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
        prefix_rows: list[list[int]] | None = None,
        prefix_tokens: torch.Tensor | None = None,
        prefix_lengths: torch.Tensor | None = None,
    ) -> IntermediateRoundState:
        bundle = self._forward_hidden_states(
            base_target_token_ids=base_target_token_ids,
            base_target_positions=base_target_positions,
            base_target_hidden_states=base_target_hidden_states,
            base_next_token_ids=base_next_token_ids,
            base_common_attn_metadata=base_common_attn_metadata,
            base_num_rejected_tokens_gpu=base_num_rejected_tokens_gpu,
            prefix_rows=prefix_rows,
            prefix_tokens=prefix_tokens,
            prefix_lengths=prefix_lengths,
        )
        return IntermediateRoundState(
            hidden_bundle=bundle,
            verification_metadata={"bootstrap_source": "intermediate_model"},
            frontier_metadata=[{"row": i, "provisional": True} for i in range(bundle.batch_size)],
            bootstrap_complete=True,
        )
