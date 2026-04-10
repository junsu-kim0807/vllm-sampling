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
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.spec_stage_ops import (
    build_family_flatten_order,
    collapse_family_paths_to_origin,
    collapse_family_tree_sampled_to_family_paths,
    get_target_verification_accepted_draft_prefix_lens,
    run_verify_stage,
)
from vllm.v1.spec_decode.spec_stage_runtime import (
    DitRoundDecision,
    DitRoundProposal,
    DitRoundVerification,
    FamilyTreeBundle,
    HybridProposalBundle,
    PivotExpandedTreePlan,
    PivotExpansionFamily,
    PivotExpansionPlan,
    PivotTreeFamily,
)
from vllm.v1.spec_decode.staged_delegate_factory import (
    PivotStagedDelegates,
    build_pivot_staged_delegates,
)
from vllm.v1.spec_decode.spec_stage_utils import slice_sampling_metadata_for_subbatch
from vllm.v1.spec_decode.utils import create_vllm_config_for_draft_model


def _chain_spec_token_tree(num_tokens: int) -> str:
    return str([(i + 1) * (0,) for i in range(num_tokens)])


def _vllm_as_plain_draft(base: VllmConfig, *, length: int) -> VllmConfig:
    spec = base.speculative_config
    assert spec is not None
    # Pivot __post_init__ sets prompt_lookup_* to 0; pydantic rejects 0 on replace().
    new_spec = replace(
        spec,
        method="draft_model",
        num_speculative_tokens=length,
        speculative_token_tree=_chain_spec_token_tree(length),
        prompt_lookup_min=1,
        prompt_lookup_max=1,
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
        prompt_lookup_min=1,
        prompt_lookup_max=1,
    )
    return replace(base, speculative_config=new_spec)


def _vllm_as_eagle_head(
    base: VllmConfig,
    *,
    length: int,
    use_eagle_tree: bool,
) -> VllmConfig:
    spec = base.speculative_config
    assert spec is not None
    new_spec = replace(
        spec,
        method="eagle3",
        num_speculative_tokens=length,
        speculative_token_tree=(
            spec.speculative_token_tree
            if use_eagle_tree and spec.speculative_token_tree is not None
            else _chain_spec_token_tree(length)
        ),
        prompt_lookup_min=1,
        prompt_lookup_max=1,
    )
    return replace(base, speculative_config=new_spec)


def _collapse_draft_rows_for_scheduler(
    out_exp: torch.Tensor,
    plan: PivotExpansionPlan | None,
    origin_batch_size: int,
) -> torch.Tensor:
    if plan is None:
        return out_exp
    cap = int(out_exp.shape[1])
    out = torch.full(
        (origin_batch_size, cap),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=out_exp.device,
    )
    for origin in range(origin_batch_size):
        row_idx = next(
            idx for idx, orig in enumerate(plan.expanded_to_origin) if orig == origin
        )
        out[origin].copy_(out_exp[row_idx])
    return out


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
        mode = spec.get_pivot_runtime_mode()
        self._L = int(spec.num_speculative_tokens)
        self._topk_selection = int(spec.pivot_topk_selection)
        self._expansion_pct = float(spec.pivot_expansion_pct)
        self._pivot_mode = mode
        self._pivot_spechive = (
            mode.verification_pipeline in (
                "intermediate_then_target",
                "intermediate_tree_then_target_tree",
            )
        )
        self._pivot_use_eagle_tree = bool(spec.pivot_use_eagle_tree)
        self._pivot_spechive_num_rounds = int(spec.pivot_spechive_num_rounds)
        self._draft: DraftModelProposer | None = None
        self._eagle_head: EagleProposer | None = None
        if mode.proposal_engine == "draft_model":
            self._draft = DraftModelProposer(
                _vllm_as_plain_draft(vllm_config, length=self._L),
                device,
                runner,
            )
        else:
            self._eagle_head = EagleProposer(
                _vllm_as_eagle_head(
                    vllm_config,
                    length=self._L,
                    use_eagle_tree=self._pivot_use_eagle_tree,
                ),
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
        self._pending_tree_plan: PivotExpandedTreePlan | None = None
        self._active_pivot_expansion_plan: PivotExpansionPlan | None = None
        self._is_waiting_for_target_collapse: bool = False
        # Incremented each target `propose()`; inner I-round count is
        # `pivot_spechive_num_rounds` inside GpuModelRunner.run_hierarchical_*.
        self._num_intermediate_rounds_since_target: int = 0
        self._pending_pivot_bundle_metadata: dict[str, object] | None = None
        self._staged_delegates: PivotStagedDelegates = build_pivot_staged_delegates(
            proposal_engine_kind=mode.proposal_engine,
            draft_delegate=self._draft,
            eagle_delegate=self._eagle_head,
            intermediate_delegate=self._intermediate,
            needs_intermediate_provider=self._pivot_spechive,
        )

    def should_force_target_verification_round(self) -> bool:
        """Hook for future outer-step policy. Inner D=>I rounds are fixed by config."""
        return False

    def _main_delegate(self):
        if self._pivot_mode.proposal_engine == "draft_model":
            assert self._draft is not None
            return self._draft
        assert self._eagle_head is not None
        return self._eagle_head

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
        if self._eagle_head is not None:
            self._eagle_head.load_model(target_model)
        self._intermediate.load_model(target_model)

    def initialize_attn_backend(self, kv_cache_config, kernel_block_sizes=None) -> None:
        if self._draft is not None:
            self._draft.initialize_attn_backend(kv_cache_config, kernel_block_sizes)
        if self._eagle_head is not None:
            self._eagle_head.initialize_attn_backend(kv_cache_config, kernel_block_sizes)
        self._intermediate.initialize_attn_backend(kv_cache_config, kernel_block_sizes)

    def initialize_cudagraph_keys(self, cudagraph_mode) -> None:
        if self._draft is not None:
            self._draft.initialize_cudagraph_keys(cudagraph_mode)
        if self._eagle_head is not None:
            self._eagle_head.initialize_cudagraph_keys(cudagraph_mode)
        self._intermediate.initialize_cudagraph_keys(cudagraph_mode)

    def dummy_run(self, *args, **kwargs):
        if self._draft is not None:
            self._draft.dummy_run(*args, **kwargs)
        if self._eagle_head is not None:
            self._eagle_head.dummy_run(*args, **kwargs)
        self._intermediate.dummy_run(*args, **kwargs)

    def validate_same_kv_cache_group(self, kv_cache_config) -> None:
        if self._draft is not None:
            self._draft.validate_same_kv_cache_group(kv_cache_config)
        if self._eagle_head is not None:
            self._eagle_head.validate_same_kv_cache_group(kv_cache_config)
        self._intermediate.validate_same_kv_cache_group(kv_cache_config)

    def clear_draft_probs(self) -> None:
        if self._draft is not None:
            self._draft.clear_draft_probs()
        if self._eagle_head is not None:
            self._eagle_head.clear_draft_probs()
        self._intermediate.clear_draft_probs()

    def take_pending_spechive_bundle(self) -> HybridProposalBundle | None:
        bundle = self._pending_hybrid_bundle
        self._pending_hybrid_bundle = None
        return bundle

    def discard_pending_hierarchical_verification_state(self) -> None:
        self._pending_hybrid_bundle = None
        self._active_pivot_expansion_plan = None
        self._pending_tree_plan = None
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

    def _select_low_confidence_origin_rows(self, pivot_probs: torch.Tensor) -> list[int]:
        return self._select_low_confidence_indices(pivot_probs)

    def _build_expanded_root_families(
        self,
        *,
        expansion_plan: PivotExpansionPlan,
    ) -> list[PivotTreeFamily]:
        families: list[PivotTreeFamily] = []
        row_meta: dict[int, tuple[int, int]] = {}
        for fam in expansion_plan.families:
            for local_idx, row_idx in enumerate(fam.expanded_rows):
                row_meta[int(row_idx)] = (
                    int(fam.candidate_ranks[local_idx]),
                    int(fam.first_token_ids[local_idx]),
                )
        for family_id, origin_row in enumerate(expansion_plan.expanded_to_origin):
            root_rank, root_token_id = row_meta.get(int(family_id), (0, 0))
            families.append(
                PivotTreeFamily(
                    origin_row=int(origin_row),
                    family_id=int(family_id),
                    root_rank=int(root_rank),
                    root_token_id=int(root_token_id),
                    node_row_start=-1,
                    node_row_end=-1,
                )
            )
        return families

    def _build_family_tree_bundle(
        self,
        *,
        proposal_tokens: torch.Tensor,
        expansion_plan: PivotExpansionPlan,
    ) -> FamilyTreeBundle | None:
        if not self._pivot_use_eagle_tree or self._eagle_head is None:
            return None
        template = self._eagle_head.get_tree_template()
        families = self._build_expanded_root_families(expansion_plan=expansion_plan)
        if not families:
            return None
        flat_plan = build_family_flatten_order(
            families=families,
            template=template,
            origin_batch_size=max(expansion_plan.expanded_to_origin) + 1,
        )
        return FamilyTreeBundle(
            plan=flat_plan,
            token_ids=proposal_tokens.reshape(-1).to(torch.int32),
            proposal_probs=None,
            reduced_prefix_rows=None,
        )

    def _flatten_family_trees_for_verification(
        self,
        *,
        proposal_tokens: torch.Tensor,
        expansion_plan: PivotExpansionPlan,
    ) -> tuple[torch.Tensor, PivotExpandedTreePlan | None]:
        bundle = self._build_family_tree_bundle(
            proposal_tokens=proposal_tokens, expansion_plan=expansion_plan
        )
        if bundle is None:
            return proposal_tokens, None
        return proposal_tokens, bundle.plan

    def _interpret_family_verification_output(
        self,
        *,
        sampled_token_ids: torch.Tensor,
        plan: PivotExpandedTreePlan,
        num_draft_tokens: list[int] | None = None,
    ):
        return collapse_family_tree_sampled_to_family_paths(
            sampled_token_ids,
            plan=plan,
            num_draft_tokens=num_draft_tokens,
        )

    def _collapse_families_to_origin(
        self,
        *,
        sampled_token_ids: torch.Tensor,
        plan: PivotExpandedTreePlan,
        reduced,
    ) -> torch.Tensor:
        collapsed, _ = collapse_family_paths_to_origin(
            sampled_token_ids,
            plan=plan,
            reduced=reduced,
        )
        return collapsed

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

    def _resolve_hidden_states_for_proposal(
        self,
        *,
        base_target_token_ids: torch.Tensor,
        base_target_positions: torch.Tensor,
        base_target_hidden_states: torch.Tensor,
        base_next_token_ids: torch.Tensor,
        base_common_attn_metadata: CommonAttentionMetadata,
        base_num_rejected_tokens_gpu: torch.Tensor | None,
        prefix_rows: list[list[int]],
    ) -> torch.Tensor:
        if self._pivot_mode.proposal_engine != "eagle3_head":
            return base_target_hidden_states
        if self._pivot_mode.hidden_state_source == "target":
            return base_target_hidden_states
        provider = self._staged_delegates.hidden_state_provider
        assert provider is not None, (
            "pivot eagle3_head + intermediate pipeline requires "
            "IntermediateModelStateProvider."
        )
        round_state = provider.bootstrap_from_current_prefix(
            base_target_token_ids=base_target_token_ids,
            base_target_positions=base_target_positions,
            base_target_hidden_states=base_target_hidden_states,
            base_next_token_ids=base_next_token_ids,
            base_common_attn_metadata=base_common_attn_metadata,
            base_num_rejected_tokens_gpu=base_num_rejected_tokens_gpu,
            prefix_rows=prefix_rows,
        )
        assert round_state.bootstrap_complete and round_state.hidden_bundle is not None, (
            "Intermediate bootstrap must produce hidden state bundle before staged proposal."
        )
        return round_state.hidden_bundle.hidden_states

    def _slice_origin_request_inputs(
        self,
        *,
        origin_row: int,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, CommonAttentionMetadata]:
        qsl = common_attn_metadata.query_start_loc
        start = int(qsl[origin_row].item())
        end = int(qsl[origin_row + 1].item())
        tok = target_token_ids[start:end]
        hid = target_hidden_states[start:end]
        if target_positions.dim() == 1:
            pos = target_positions[start:end]
        else:
            pos = target_positions[:, start:end]
        nxt = next_token_ids[origin_row : origin_row + 1]
        qsl_one = torch.tensor([0, end - start], dtype=qsl.dtype, device=qsl.device)
        cad_one = common_attn_metadata.replace(
            query_start_loc=qsl_one,
            query_start_loc_cpu=qsl_one.detach().cpu(),
            seq_lens=common_attn_metadata.seq_lens[origin_row : origin_row + 1].clone(),
            block_table_tensor=common_attn_metadata.block_table_tensor[
                origin_row : origin_row + 1
            ].clone(),
            slot_mapping=common_attn_metadata.slot_mapping[start:end].clone(),
            logits_indices_padded=(
                common_attn_metadata.logits_indices_padded[
                    origin_row : origin_row + 1
                ].clone()
                if common_attn_metadata.logits_indices_padded is not None
                else None
            ),
            dcp_local_seq_lens=(
                common_attn_metadata.dcp_local_seq_lens[origin_row : origin_row + 1].clone()
                if common_attn_metadata.dcp_local_seq_lens is not None
                else None
            ),
            num_reqs=1,
            num_actual_tokens=int(end - start),
            max_query_len=int(end - start),
            max_seq_len=int(common_attn_metadata.seq_lens[origin_row].item()),
            _seq_lens_cpu=None,
            _num_computed_tokens_cpu=None,
            _num_computed_tokens_cache=None,
        )
        return tok, pos, hid, nxt, cad_one

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
        if tail_len > 0:
            pivot_list = pivots.view(batch_size).tolist()
            full_prefix_rows = [
                [*base_prefix_rows[b], int(tok)] for b, tok in enumerate(pivot_list)
            ]
            tail_sm = slice_sampling_metadata_for_subbatch(
                sampling_metadata,
                list(range(batch_size)),
                provisional_prefix_rows=full_prefix_rows,
                sampled_ids_only=True,
            )
            proposal_hidden_states = self._resolve_hidden_states_for_proposal(
                base_target_token_ids=base_target_token_ids,
                base_target_positions=base_target_positions,
                base_target_hidden_states=base_target_hidden_states,
                base_next_token_ids=base_next_token_ids,
                base_common_attn_metadata=base_common_attn_metadata,
                base_num_rejected_tokens_gpu=base_num_rejected_tokens_gpu,
                prefix_rows=full_prefix_rows,
            )
            tail_rows, tail_probs = _propose_chunk_from_prefix(
                self._main_delegate(),
                cad=base_common_attn_metadata,
                target_token_ids=base_target_token_ids,
                target_positions=base_target_positions,
                target_hidden_states=proposal_hidden_states,
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
        proposal_hidden_states = self._resolve_hidden_states_for_proposal(
            base_target_token_ids=base_target_token_ids,
            base_target_positions=base_target_positions,
            base_target_hidden_states=base_target_hidden_states,
            base_next_token_ids=base_next_token_ids,
            base_common_attn_metadata=base_common_attn_metadata,
            base_num_rejected_tokens_gpu=base_num_rejected_tokens_gpu,
            prefix_rows=base_prefix_rows,
        )
        pivots, pivot_probs = _propose_chunk_from_prefix(
            self._main_delegate(),
            cad=base_common_attn_metadata,
            target_token_ids=base_target_token_ids,
            target_positions=base_target_positions,
            target_hidden_states=proposal_hidden_states,
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

    def _propose_via_eagle_tree(
        self,
        *,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
        use_draft_probs: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, PivotExpansionPlan | None]:
        batch_size = int(common_attn_metadata.batch_size())
        assert self._eagle_head is not None
        base_prefix_rows = [[] for _ in range(batch_size)]
        proposal_hidden_states = self._resolve_hidden_states_for_proposal(
            base_target_token_ids=target_token_ids,
            base_target_positions=target_positions,
            base_target_hidden_states=target_hidden_states,
            base_next_token_ids=next_token_ids,
            base_common_attn_metadata=common_attn_metadata,
            base_num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            prefix_rows=base_prefix_rows,
        )
        pivots, pivot_probs = _propose_chunk_from_prefix(
            self._eagle_head,
            cad=common_attn_metadata,
            target_token_ids=target_token_ids,
            target_positions=target_positions,
            target_hidden_states=proposal_hidden_states,
            next_token_ids=next_token_ids,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            prefix_rows=base_prefix_rows,
            chunk_len=1,
            sampling_metadata=sampling_metadata,
            use_draft_probs=True,
        )
        pivots = pivots.to(torch.int32)[:, :1]
        chosen_pivots, _, expansion_plan = self._build_pivot_expansion_plan(
            initial_pivots=pivots,
            pivot_probs=pivot_probs,
            enable_topk_expansion=True,
        )
        if expansion_plan is not None:
            self._active_pivot_expansion_plan = expansion_plan
            if self._pivot_spechive:
                self._is_waiting_for_target_collapse = True
            rows_per_family: list[torch.Tensor] = []
            for fam_row, origin_row in enumerate(expansion_plan.expanded_to_origin):
                tok, pos, hid, nxt, cad = self._slice_origin_request_inputs(
                    origin_row=int(origin_row),
                    target_token_ids=target_token_ids,
                    target_positions=target_positions,
                    target_hidden_states=proposal_hidden_states,
                    next_token_ids=next_token_ids,
                    common_attn_metadata=common_attn_metadata,
                )
                sm = slice_sampling_metadata_for_subbatch(
                    sampling_metadata,
                    [int(origin_row)],
                    sampled_ids_only=True,
                )
                drafted = self._eagle_head.propose_tree_from_prefix(
                    target_token_ids=tok,
                    target_positions=pos,
                    target_hidden_states=hid,
                    next_token_ids=nxt,
                    common_attn_metadata=cad,
                    sampling_metadata=sm,
                    prefix_rows=None,
                    root_token_override=chosen_pivots[fam_row : fam_row + 1, 0],
                    num_rejected_tokens_gpu=num_rejected_tokens_gpu,
                )
                rows_per_family.append(drafted[0].to(torch.int32))
            rows = torch.stack(rows_per_family, dim=0)
            assert all(len(r) == 0 for r in base_prefix_rows), (
                "pivot tree root expansion must only happen at root prefix."
            )
            assert len(expansion_plan.expanded_to_origin) == int(rows.shape[0]), (
                "pivot tree expanded family count must match proposal rows."
            )
        else:
            rows = self._eagle_head.propose_tree_from_prefix(
                target_token_ids=target_token_ids,
                target_positions=target_positions,
                target_hidden_states=proposal_hidden_states,
                next_token_ids=next_token_ids,
                common_attn_metadata=common_attn_metadata,
                sampling_metadata=sampling_metadata,
                prefix_rows=None,
                root_token_override=chosen_pivots.view(-1),
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            )
        tree_template = self._eagle_head.get_tree_template()
        assert int(rows.shape[1]) == int(tree_template.num_nodes), (
            f"pivot tree proposal width {rows.shape[1]} "
            f"must match tree template nodes {tree_template.num_nodes}."
        )
        probs_flat = None
        if use_draft_probs and expansion_plan is None:
            probs_flat = getattr(self._eagle_head, "last_draft_probs_flat", None)
        # Expanded tree families currently disable stochastic proposal probs.
        if expansion_plan is not None:
            probs_flat = None
        return rows.to(torch.int32), probs_flat, expansion_plan

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
            # Until branch-conditioned proposal math is fully implemented,
            # disable stochastic draft probs for expanded pivot families.
            probs_flat = None
        probs = None
        if probs_flat is not None and probs_flat.numel() > 0:
            probs = probs_flat.view(rows.shape[0], rows.shape[1], probs_flat.shape[-1])
        tree_plan = None
        if self._pivot_use_eagle_tree and expansion_plan is not None:
            _, tree_plan = self._flatten_family_trees_for_verification(
                proposal_tokens=rows,
                expansion_plan=expansion_plan,
            )
        return DitRoundProposal(
            tokens=rows,
            probs=probs,
            expansion_plan=expansion_plan,
            tree_plan=tree_plan,
        )

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
        verifier_hidden_states = base_target_hidden_states
        verifier_round_state = None
        if self._pivot_mode.verification_pipeline in (
            "intermediate_then_target",
            "intermediate_tree_then_target_tree",
        ):
            provider = self._staged_delegates.hidden_state_provider
            assert provider is not None
            round_state = provider.bootstrap_from_current_prefix(
                base_target_token_ids=base_target_token_ids,
                base_target_positions=base_target_positions,
                base_target_hidden_states=base_target_hidden_states,
                base_next_token_ids=base_next_token_ids,
                base_common_attn_metadata=base_common_attn_metadata,
                base_num_rejected_tokens_gpu=base_num_rejected_tokens_gpu,
                prefix_rows=prefix_rows,
            )
            assert round_state.hidden_bundle is not None
            verifier_round_state = round_state
            verifier_hidden_states = round_state.hidden_bundle.hidden_states
        if (
            self._pivot_mode.proposal_engine == "eagle3_head"
            and self._pivot_mode.verification_pipeline
            == "intermediate_tree_then_target_tree"
        ):
            assert (
                verifier_round_state is not None
                and verifier_round_state.hidden_bundle is not None
            ), "pivot tree intermediate hidden bundle must be present."
            assert int(verifier_hidden_states.shape[0]) >= int(candidate_tokens.shape[0]), (
                "pivot tree intermediate hidden batch must cover family rows: "
                f"hidden_rows={int(verifier_hidden_states.shape[0])}, "
                f"families={int(candidate_tokens.shape[0])}"
            )
        logits_flat, bonus_logits = _verify_chunk_with_prefix(
            self._intermediate,
            cad=base_common_attn_metadata,
            target_token_ids=base_target_token_ids,
            target_positions=base_target_positions,
            target_hidden_states=verifier_hidden_states,
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
        if proposal.tree_plan is not None:
            tree_nodes = int(proposal.tree_plan.template.num_nodes)
            assert chunk_len == tree_nodes, (
                "pivot tree intermediate round width must match tree nodes: "
                f"tokens_shape={tuple(proposal.tokens.shape)}, tree_nodes={tree_nodes}"
            )
            assert len(proposal.tree_plan.families) == batch_size, (
                "pivot tree family count must match proposal batch: "
                f"families={len(proposal.tree_plan.families)}, batch={batch_size}"
            )
        if proposal.expansion_plan is not None:
            assert batch_size == proposal.expansion_plan.expanded_batch_size, (
                "DitRoundProposal.tokens batch must match expansion_plan.expanded_batch_size"
            )
        draft_flat = proposal.tokens.reshape(-1).to(torch.int32)
        if proposal.tree_plan is not None:
            num_draft_tokens = [
                int(end - start) for (start, end) in proposal.tree_plan.family_flat_spans
            ]
            batch_size = len(num_draft_tokens)
        else:
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
            max_spec_len=max(num_draft_tokens) if num_draft_tokens else chunk_len,
            verifier_logits_flat=verification.logits_flat,
            bonus_logits=verification.bonus_logits,
            sampling_metadata=sampling_metadata,
            vocab_size=vocab_size,
            include_processed_probs=use_draft_probs,
        )
        emitted_prob_rows: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
        if use_draft_probs and target_probs_flat.numel() > 0 and proposal.tree_plan is None:
            per_req_probs = target_probs_flat.view(batch_size, chunk_len, vocab_size)
            for b in range(batch_size):
                emitted_len = len(emitted_rows[b])
                clipped_len = min(emitted_len, chunk_len)
                emitted_prob_rows[b] = [per_req_probs[b, i] for i in range(clipped_len)]
                if emitted_len > chunk_len and bonus_probs.numel() > 0:
                    emitted_prob_rows[b].append(bonus_probs[b].to(torch.float32))
        accepted_lens = get_target_verification_accepted_draft_prefix_lens(
            stage_out.sampled_token_ids,
            num_draft_tokens,
            placeholder_token_id=PLACEHOLDER_TOKEN_ID,
        )
        if proposal.tree_plan is not None:
            reduced = collapse_family_tree_sampled_to_family_paths(
                stage_out.sampled_token_ids,
                plan=proposal.tree_plan,
                num_draft_tokens=num_draft_tokens,
            )
            emitted_rows = reduced.accepted_rows
            accepted_lens = reduced.accepted_lens
            # Tree-mode stochastic per-node proposal probs are not lossless yet.
            emitted_prob_rows = [[] for _ in range(len(emitted_rows))]
        return DitRoundDecision(
            emitted_rows=emitted_rows,
            emitted_prob_rows=emitted_prob_rows,
            accepted_lens=accepted_lens,
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
            if self._pivot_use_eagle_tree:
                out, bundle = self.runner.run_hierarchical_tree_verification_rounds(
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
            else:
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
            self._pending_tree_plan = bundle.tree_plan
            self._num_intermediate_rounds_since_target += 1
            self._pending_pivot_bundle_metadata = {
                "expansion_plan": self._active_pivot_expansion_plan,
                "is_waiting_for_target_collapse": self._is_waiting_for_target_collapse,
                "num_intermediate_rounds_since_target": self._num_intermediate_rounds_since_target,
            }
            self.clear_draft_probs()
            return out

        batch_size = int(common_attn_metadata.batch_size())
        if self._pivot_use_eagle_tree:
            out, draft_probs_flat, expansion_plan = self._propose_via_eagle_tree(
                target_token_ids=target_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                next_token_ids=next_token_ids,
                common_attn_metadata=common_attn_metadata,
                sampling_metadata=sampling_metadata,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
                use_draft_probs=use_draft_probs,
            )
        else:
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
            _, tree_plan = self._flatten_family_trees_for_verification(
                proposal_tokens=out,
                expansion_plan=expansion_plan,
            )
            self._pending_tree_plan = tree_plan
            self._pending_hybrid_bundle = dataclass_replace(
                self._pending_hybrid_bundle,
                expansion_plan=expansion_plan,
                tree_plan=tree_plan,
            )
        else:
            self._active_pivot_expansion_plan = None
            self._is_waiting_for_target_collapse = False
            self._pending_tree_plan = None
        self._pending_pivot_bundle_metadata = {
            "expansion_plan": self._active_pivot_expansion_plan,
            "is_waiting_for_target_collapse": self._is_waiting_for_target_collapse,
            "num_intermediate_rounds_since_target": self._num_intermediate_rounds_since_target,
        }
        self.clear_draft_probs()
        return _collapse_draft_rows_for_scheduler(out, expansion_plan, batch_size)

    def on_target_verification(
        self,
        *,
        bundle: HybridProposalBundle,
        sampled_token_ids: torch.Tensor,
    ) -> None:
        del bundle, sampled_token_ids
        self._active_pivot_expansion_plan = None
        self._is_waiting_for_target_collapse = False
        self._pending_tree_plan = None
        self._num_intermediate_rounds_since_target = 0
        self._pending_pivot_bundle_metadata = None
