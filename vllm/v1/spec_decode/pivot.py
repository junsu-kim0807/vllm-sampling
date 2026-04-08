# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone pivot proposer for speculative decoding.

Pivot method semantics:
- draft model emits the first token candidate (with optional top-k expansion).
- draft model emits the remaining tail tokens conditioned on recovery+first-token.
- intermediate model is used for staged verification rounds.
- target model verifies the concatenated candidate bundle.
"""

from __future__ import annotations

import math
from dataclasses import replace as dataclass_replace

import torch
import torch.nn as nn
from typing_extensions import override

from vllm.config import VllmConfig, replace
from vllm.model_executor.model_loader import get_model
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID
from vllm.v1.spec_decode.adaptive_cascade import (
    _propose_chunk_from_prefix,
    _verify_chunk_with_prefix,
)
from vllm.v1.spec_decode.adaptive_spechive import (
    _build_hybrid_bundle_from_rows,
    _flatten_prob_rows_for_output,
)
from vllm.v1.spec_decode.draft_model import DraftModelProposer
from vllm.v1.spec_decode.spec_stage_ops import (
    get_target_verification_accepted_draft_prefix_lens,
    run_verify_stage,
)
from vllm.v1.spec_decode.spec_stage_runtime import (
    DitRoundDecision,
    DitRoundProposal,
    DitRoundVerification,
    HybridProposalBundle,
    PivotExpansionFamily,
    PivotExpansionPlan,
)
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
        self._topk_selection = int(spec.pivot_topk_selection)
        self._expansion_pct = float(spec.pivot_expansion_pct)
        self._pivot_spechive = bool(spec.pivot_spechive)
        self._pivot_spechive_num_rounds = int(spec.pivot_spechive_num_rounds)
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
        # Keep parity with AdaptiveSpechiveProposer contract used by
        # run_hierarchical_verification_rounds().
        self._inter_dit: IntermediatePivotModelProposer = self._intermediate
        self._pending_hybrid_bundle: HybridProposalBundle | None = None
        self._active_pivot_expansion_plan: PivotExpansionPlan | None = None
        self._is_waiting_for_target_collapse: bool = False
        # Incremented each target `propose()`; inner I-round count is
        # `pivot_spechive_num_rounds` inside GpuModelRunner.run_hierarchical_*.
        self._num_intermediate_rounds_since_target: int = 0
        self._pending_pivot_bundle_metadata: dict[str, object] | None = None

    def should_force_target_verification_round(self) -> bool:
        """Hook for future outer-step policy. Inner D=>I rounds are fixed by config."""
        return False

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
        self._active_pivot_expansion_plan = None
        self._is_waiting_for_target_collapse = False
        self._pending_pivot_bundle_metadata = None

    def _select_low_confidence_indices(self, pivot_probs: torch.Tensor) -> list[int]:
        batch_size = int(pivot_probs.shape[0])
        if batch_size <= 0:
            return []
        num_expand = int(math.ceil(batch_size * self._expansion_pct))
        num_expand = min(max(num_expand, 0), batch_size)
        if num_expand == 0:
            return []
        top1_prob = pivot_probs[:, 0, :].amax(dim=-1)
        chosen = torch.topk(top1_prob, k=num_expand, largest=False).indices
        return [int(i) for i in chosen.tolist()]

    def _build_pivot_expansion_plan(
        self,
        *,
        initial_pivots: torch.Tensor,
        pivot_probs: torch.Tensor,
        enable_topk_expansion: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, PivotExpansionPlan | None]:
        """Build B' pivot rows and per-row first-step probs (same q(·) as origin)."""
        batch_size = int(initial_pivots.shape[0])
        if (
            not enable_topk_expansion
            or self._topk_selection <= 1
            or batch_size == 0
        ):
            ep = initial_pivots.to(torch.int32)
            eprob = (
                pivot_probs[:, :1, :].clone()
                if pivot_probs is not None
                else None
            )
            return ep, eprob, None

        selected_set = set(self._select_low_confidence_indices(pivot_probs))
        if not selected_set:
            ep = initial_pivots.to(torch.int32)
            eprob = (
                pivot_probs[:, :1, :].clone()
                if pivot_probs is not None
                else None
            )
            return ep, eprob, None

        expanded_to_origin: list[int] = []
        families: list[PivotExpansionFamily] = []
        expanded_row = 0
        pivot_cols: list[torch.Tensor] = []
        for origin in range(batch_size):
            if origin not in selected_set:
                expanded_to_origin.append(origin)
                pivot_cols.append(initial_pivots[origin : origin + 1].to(torch.int32))
                expanded_row += 1
                continue

            k = min(self._topk_selection, int(pivot_probs.shape[-1]))
            cand_ids = torch.topk(pivot_probs[origin, 0], k=k, largest=True).indices.tolist()
            cand_ids = [int(t) for t in cand_ids]
            cand_probs = [float(pivot_probs[origin, 0, t].item()) for t in cand_ids]
            expanded_rows = list(range(expanded_row, expanded_row + len(cand_ids)))
            expanded_to_origin.extend([origin] * len(cand_ids))
            families.append(
                PivotExpansionFamily(
                    origin_row=origin,
                    expanded_rows=expanded_rows,
                    candidate_ranks=list(range(len(cand_ids))),
                    first_token_ids=cand_ids,
                    first_token_probs=cand_probs,
                )
            )
            for tid in cand_ids:
                pivot_cols.append(
                    torch.tensor([[tid]], dtype=torch.int32, device=initial_pivots.device)
                )
            expanded_row += len(cand_ids)

        expanded_pivots = torch.cat(pivot_cols, dim=0)
        plan = PivotExpansionPlan(
            expanded_to_origin=expanded_to_origin,
            families=families,
            expanded_batch_size=len(expanded_to_origin),
        )
        idx = torch.tensor(expanded_to_origin, device=pivot_probs.device, dtype=torch.long)
        expanded_probs = pivot_probs[idx, :1, :].clone()
        return expanded_pivots, expanded_probs, plan

    @staticmethod
    def _expand_prefix_rows_for_plan(
        base_prefix_rows: list[list[int]],
        plan: PivotExpansionPlan,
    ) -> list[list[int]]:
        return [list(base_prefix_rows[o]) for o in plan.expanded_to_origin]

    def _expand_sampling_metadata_for_plan(
        self,
        sampling_metadata: SamplingMetadata,
        plan: PivotExpansionPlan,
        expanded_prefix_rows: list[list[int]],
    ) -> SamplingMetadata:
        return slice_sampling_metadata_for_subbatch(
            sampling_metadata,
            plan.expanded_to_origin,
            provisional_prefix_rows=expanded_prefix_rows,
            sampled_ids_only=True,
        )

    def _compose_tokens_from_pivots(
        self,
        *,
        base_target_token_ids: torch.Tensor,
        base_target_positions: torch.Tensor,
        base_target_hidden_states: torch.Tensor,
        base_next_token_ids: torch.Tensor,
        base_common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        base_num_rejected_tokens_gpu: torch.Tensor | None,
        base_prefix_rows: list[list[int]],
        pivots: torch.Tensor,
        chunk_len: int,
        use_draft_probs: bool,
        pivot_probs: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size = int(pivots.shape[0])
        out = torch.full(
            (batch_size, chunk_len),
            PLACEHOLDER_TOKEN_ID,
            dtype=torch.int32,
            device=base_target_token_ids.device,
        )
        out[:, :1] = pivots.to(torch.int32)

        tail_len = max(0, chunk_len - 1)
        tail_probs: torch.Tensor | None = None
        if tail_len > 0 and self._draft is not None:
            pivot_list = pivots.view(batch_size).tolist()
            only_pivot_rows = [[int(tok)] for tok in pivot_list]
            full_prefix_rows = [
                [*base_prefix_rows[b], int(tok)] for b, tok in enumerate(pivot_list)
            ]
            tail_sm = slice_sampling_metadata_for_subbatch(
                sampling_metadata,
                list(range(batch_size)),
                provisional_prefix_rows=only_pivot_rows,
                sampled_ids_only=True,
            )
            tail_rows, tail_probs = _propose_chunk_from_prefix(
                self._draft,
                cad=base_common_attn_metadata,
                target_token_ids=base_target_token_ids,
                target_positions=base_target_positions,
                target_hidden_states=base_target_hidden_states,
                next_token_ids=base_next_token_ids,
                num_rejected_tokens_gpu=base_num_rejected_tokens_gpu,
                prefix_rows=full_prefix_rows,
                chunk_len=tail_len,
                sampling_metadata=tail_sm,
                use_draft_probs=use_draft_probs,
            )
            out[:, 1 : 1 + tail_len] = tail_rows

        draft_probs_flat: torch.Tensor | None = None
        if use_draft_probs:
            if pivot_probs is not None and (tail_len == 0 or tail_probs is not None):
                prob_rows: list[list[torch.Tensor]] = []
                for b in range(batch_size):
                    row = [pivot_probs[b, 0]]
                    if tail_len > 0 and tail_probs is not None:
                        row.extend([tail_probs[b, j] for j in range(tail_len)])
                    prob_rows.append(row)
                draft_probs_flat = _flatten_prob_rows_for_output(prob_rows, out)
        return out, draft_probs_flat

    def _propose_chunk_with_optional_topk(
        self,
        *,
        base_target_token_ids: torch.Tensor,
        base_target_positions: torch.Tensor,
        base_target_hidden_states: torch.Tensor,
        base_next_token_ids: torch.Tensor,
        base_common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        base_num_rejected_tokens_gpu: torch.Tensor | None,
        base_prefix_rows: list[list[int]],
        chunk_len: int,
        use_draft_probs: bool,
        enable_topk_expansion: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, PivotExpansionPlan | None]:
        need_confidence_probs = enable_topk_expansion and self._topk_selection > 1
        first_step_proposer = self._draft if self._draft is not None else self._intermediate
        pivots, pivot_probs = _propose_chunk_from_prefix(
            first_step_proposer,
            cad=base_common_attn_metadata,
            target_token_ids=base_target_token_ids,
            target_positions=base_target_positions,
            target_hidden_states=base_target_hidden_states,
            next_token_ids=base_next_token_ids,
            num_rejected_tokens_gpu=base_num_rejected_tokens_gpu,
            prefix_rows=base_prefix_rows,
            chunk_len=1,
            sampling_metadata=sampling_metadata,
            use_draft_probs=(use_draft_probs or need_confidence_probs),
        )
        pivots = pivots.to(torch.int32)[:, :1]
        chosen_pivots, pivot_probs_rows, expansion_plan = self._build_pivot_expansion_plan(
            initial_pivots=pivots,
            pivot_probs=pivot_probs,
            enable_topk_expansion=enable_topk_expansion,
        )
        if expansion_plan is not None:
            self._active_pivot_expansion_plan = expansion_plan
            if self._pivot_spechive:
                self._is_waiting_for_target_collapse = True
        elif not self._pivot_spechive:
            self._active_pivot_expansion_plan = None

        if expansion_plan is not None:
            exp_prefix = self._expand_prefix_rows_for_plan(
                base_prefix_rows, expansion_plan
            )
            exp_sm = self._expand_sampling_metadata_for_plan(
                sampling_metadata, expansion_plan, exp_prefix
            )
        else:
            exp_prefix = base_prefix_rows
            exp_sm = sampling_metadata

        out, probs = self._compose_tokens_from_pivots(
            base_target_token_ids=base_target_token_ids,
            base_target_positions=base_target_positions,
            base_target_hidden_states=base_target_hidden_states,
            base_next_token_ids=base_next_token_ids,
            base_common_attn_metadata=base_common_attn_metadata,
            sampling_metadata=exp_sm,
            base_num_rejected_tokens_gpu=base_num_rejected_tokens_gpu,
            base_prefix_rows=exp_prefix,
            pivots=chosen_pivots,
            chunk_len=chunk_len,
            use_draft_probs=use_draft_probs,
            pivot_probs=pivot_probs_rows,
        )
        return out, probs, expansion_plan

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
        enable_topk = (
            self._topk_selection > 1
            and all(len(r) == 0 for r in prefix_rows)
            and not self._is_waiting_for_target_collapse
            and not (
                self._pivot_spechive and self._active_pivot_expansion_plan is not None
            )
        )
        rows, probs_flat, expansion_plan = self._propose_chunk_with_optional_topk(
            base_target_token_ids=base_target_token_ids,
            base_target_positions=base_target_positions,
            base_target_hidden_states=base_target_hidden_states,
            base_next_token_ids=base_next_token_ids,
            base_common_attn_metadata=base_common_attn_metadata,
            sampling_metadata=sampling_metadata,
            base_num_rejected_tokens_gpu=base_num_rejected_tokens_gpu,
            base_prefix_rows=prefix_rows,
            chunk_len=chunk_len,
            use_draft_probs=use_draft_probs,
            enable_topk_expansion=enable_topk,
        )
        if (
            expansion_plan is None
            and self._pivot_spechive
            and self._active_pivot_expansion_plan is not None
        ):
            expansion_plan = self._active_pivot_expansion_plan
        if expansion_plan is not None:
            assert rows.shape[0] == expansion_plan.expanded_batch_size, (
                f"proposal rows {rows.shape[0]} vs plan {expansion_plan.expanded_batch_size}"
            )
        probs = None
        if probs_flat is not None and probs_flat.numel() > 0:
            probs = probs_flat.view(rows.shape[0], rows.shape[1], probs_flat.shape[-1])
        return DitRoundProposal(tokens=rows, probs=probs, expansion_plan=expansion_plan)

    def verify_chunk_with_inter_verifier(
        self,
        *,
        base_target_token_ids: torch.Tensor,
        base_target_positions: torch.Tensor,
        base_target_hidden_states: torch.Tensor,
        base_next_token_ids: torch.Tensor,
        base_common_attn_metadata: CommonAttentionMetadata,
        base_num_rejected_tokens_gpu: torch.Tensor | None,
        prefix_rows: list[list[int]],
        candidate_tokens: torch.Tensor,
    ) -> DitRoundVerification:
        logits_flat, bonus_logits = _verify_chunk_with_prefix(
            self._intermediate,
            cad=base_common_attn_metadata,
            target_token_ids=base_target_token_ids,
            target_positions=base_target_positions,
            target_hidden_states=base_target_hidden_states,
            next_token_ids=base_next_token_ids,
            num_rejected_tokens_gpu=base_num_rejected_tokens_gpu,
            prefix_rows=prefix_rows,
            draft_tokens=candidate_tokens.to(torch.int32),
        )
        return DitRoundVerification(logits_flat=logits_flat, bonus_logits=bonus_logits)

    def run_inter_verification_acceptance(
        self,
        *,
        proposal: DitRoundProposal,
        verification: DitRoundVerification,
        sampling_metadata: SamplingMetadata,
        rejection_sampler,
        vocab_size: int,
        use_draft_probs: bool,
    ) -> DitRoundDecision:
        batch_size = int(proposal.tokens.shape[0])
        chunk_len = int(proposal.tokens.shape[1])
        if proposal.expansion_plan is not None:
            assert batch_size == proposal.expansion_plan.expanded_batch_size, (
                "DitRoundProposal.tokens batch must match expansion_plan.expanded_batch_size"
            )
        draft_flat = proposal.tokens.reshape(-1).to(torch.int32)
        num_draft_tokens = [chunk_len] * batch_size
        cu_num_draft_tokens = torch.cumsum(
            torch.tensor(num_draft_tokens, dtype=torch.int32, device=draft_flat.device),
            dim=0,
        )
        draft_probs_flat = (
            proposal.probs.reshape(-1, vocab_size) if proposal.probs is not None else None
        )
        emitted_rows, stage_out, target_probs_flat, bonus_probs = run_verify_stage(
            rejection_sampler,
            draft_token_ids_flat=draft_flat,
            draft_probs_flat=draft_probs_flat,
            num_draft_tokens=num_draft_tokens,
            cu_num_draft_tokens=cu_num_draft_tokens,
            max_spec_len=chunk_len,
            verifier_logits_flat=verification.logits_flat,
            bonus_logits=verification.bonus_logits,
            sampling_metadata=sampling_metadata,
            vocab_size=vocab_size,
            include_processed_probs=use_draft_probs,
        )
        emitted_prob_rows: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
        if use_draft_probs and target_probs_flat.numel() > 0:
            per_req_probs = target_probs_flat.view(batch_size, chunk_len, vocab_size)
            for b in range(batch_size):
                emitted_len = len(emitted_rows[b])
                clipped_len = min(emitted_len, chunk_len)
                emitted_prob_rows[b] = [per_req_probs[b, i] for i in range(clipped_len)]
                if emitted_len > chunk_len and bonus_probs.numel() > 0:
                    emitted_prob_rows[b].append(bonus_probs[b].to(torch.float32))
        num_draft = [chunk_len] * batch_size
        return DitRoundDecision(
            emitted_rows=emitted_rows,
            emitted_prob_rows=emitted_prob_rows,
            accepted_lens=get_target_verification_accepted_draft_prefix_lens(
                stage_out.sampled_token_ids,
                num_draft,
                placeholder_token_id=PLACEHOLDER_TOKEN_ID,
            ),
        )

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
        use_draft_probs = (
            self.vllm_config.speculative_config is not None
            and self.vllm_config.speculative_config.use_draft_probs_in_rejection
            and not sampling_metadata.all_greedy
        )
        if self._pivot_spechive:
            assert self.runner is not None
            out, bundle = self.runner.run_hierarchical_verification_rounds(
                drafter=self,  # type: ignore[arg-type]
                target_token_ids=target_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                next_token_ids=next_token_ids,
                token_indices_to_sample=token_indices_to_sample,
                common_attn_metadata=common_attn_metadata,
                sampling_metadata=sampling_metadata,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            )
            self._pending_hybrid_bundle = bundle
            self._num_intermediate_rounds_since_target += 1
            self._pending_pivot_bundle_metadata = {
                "expansion_plan": self._active_pivot_expansion_plan,
                "is_waiting_for_target_collapse": self._is_waiting_for_target_collapse,
                "num_intermediate_rounds_since_target": self._num_intermediate_rounds_since_target,
            }
            self.clear_draft_probs()
            return out

        batch_size = int(common_attn_metadata.batch_size())
        out, draft_probs_flat, expansion_plan = self._propose_chunk_with_optional_topk(
            base_target_token_ids=target_token_ids,
            base_target_positions=target_positions,
            base_target_hidden_states=target_hidden_states,
            base_next_token_ids=next_token_ids,
            base_common_attn_metadata=common_attn_metadata,
            sampling_metadata=sampling_metadata,
            base_num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            base_prefix_rows=[[] for _ in range(batch_size)],
            chunk_len=self._L,
            use_draft_probs=use_draft_probs,
            enable_topk_expansion=True,
        )
        source_stage_rows = [
            [0, *([1] * max(0, self._L - 1))] for _ in range(int(out.shape[0]))
        ]
        self._pending_hybrid_bundle = _build_hybrid_bundle_from_rows(
            out,
            mode="pivot",
            draft_probs=draft_probs_flat,
            source_stage_rows=source_stage_rows,
        )
        if expansion_plan is not None:
            self._active_pivot_expansion_plan = expansion_plan
            self._is_waiting_for_target_collapse = self._pivot_spechive
            self._pending_hybrid_bundle = dataclass_replace(
                self._pending_hybrid_bundle, expansion_plan=expansion_plan
            )
        else:
            self._active_pivot_expansion_plan = None
            self._is_waiting_for_target_collapse = False
        self._pending_pivot_bundle_metadata = {
            "expansion_plan": self._active_pivot_expansion_plan,
            "is_waiting_for_target_collapse": self._is_waiting_for_target_collapse,
            "num_intermediate_rounds_since_target": self._num_intermediate_rounds_since_target,
        }
        self.clear_draft_probs()
        return out

    def on_target_verification(
        self,
        *,
        bundle: HybridProposalBundle,
        sampled_token_ids: torch.Tensor,
    ) -> None:
        del bundle, sampled_token_ids
        self._active_pivot_expansion_plan = None
        self._is_waiting_for_target_collapse = False
        self._num_intermediate_rounds_since_target = 0
        self._pending_pivot_bundle_metadata = None
