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

from vllm.config import VllmConfig, get_layers_from_vllm_config, replace
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models import supports_multimodal
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID
from vllm.v1.spec_decode.adaptive_cascade import (
    _propose_chunk_from_prefix,
    _spechive_debug_enabled,
    _verify_chunk_with_prefix,
    verify_intermediate_chunk_with_prefix_prefab,
)
from vllm.v1.spec_decode.hybrid_bundle_utils import (
    _build_hybrid_bundle_from_rows,
    _flatten_prob_rows_for_output,
)
from vllm.v1.spec_decode.draft_model import (
    DraftModelProposer,
    PinnedDraftNamespaceDraftModelProposer,
)
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
    IntermediateRoundState,
    PivotExpandedTreePlan,
    PivotExpansionFamily,
    PivotExpansionPlan,
    PivotTreeFamily,
    RootTopKInfo,
)
from vllm.v1.spec_decode.staged_delegate_factory import (
    PivotStagedDelegates,
    build_pivot_staged_delegates,
)
from vllm.v1.spec_decode.spec_stage_utils import slice_sampling_metadata_for_subbatch
from vllm.v1.spec_decode.utils import create_vllm_config_for_draft_model

logger = init_logger(__name__)


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


def _vllm_target_as_draft_chunk(base: VllmConfig, *, length: int) -> VllmConfig:
    """VllmConfig slice for DIT prefix verification on the target model (no extra load)."""
    spec = base.speculative_config
    assert spec is not None
    assert spec.target_model_config is not None
    assert spec.target_parallel_config is not None
    tp = spec.target_parallel_config.tensor_parallel_size
    new_spec = replace(
        spec,
        method="draft_model",
        draft_model_config=spec.target_model_config,
        draft_parallel_config=spec.target_parallel_config,
        draft_tensor_parallel_size=tp,
        num_speculative_tokens=length,
        speculative_token_tree=_chain_spec_token_tree(length),
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
    if origin_batch_size <= 0:
        return out_exp[:0]

    if (
        plan.uses_fixed_capacity_packing
        and plan.origin_to_base_row is not None
        and len(plan.origin_to_base_row) >= origin_batch_size
        and all(int(plan.origin_to_base_row[o]) == o for o in range(origin_batch_size))
    ):
        return out_exp[:origin_batch_size]

    row_ids: list[int] = []
    if (
        plan.origin_to_base_row is not None
        and len(plan.origin_to_base_row) >= origin_batch_size
    ):
        row_ids = [int(plan.origin_to_base_row[o]) for o in range(origin_batch_size)]
    else:
        expanded_to_origin = plan.expanded_to_origin
        for origin in range(origin_batch_size):
            row_ids.append(
                next(
                    idx
                    for idx, orig in enumerate(expanded_to_origin)
                    if int(orig) == origin
                )
            )

    idx = torch.tensor(row_ids, dtype=torch.long, device=out_exp.device)
    return out_exp.index_select(0, idx)


def _pivot_hybrid_bundle_row_req_ids(
    runner: object | None,
    origin_batch_size: int,
    num_bundle_rows: int,
    expansion_plan: PivotExpansionPlan | None,
) -> tuple[str, ...] | None:
    """Map each proposal row to the scheduler request id for that origin row."""
    if runner is None or origin_batch_size <= 0 or num_bundle_rows <= 0:
        return None
    ib = getattr(runner, "input_batch", None)
    if ib is None:
        return None
    try:
        req = [str(ib.req_ids[i]) for i in range(origin_batch_size)]
    except (IndexError, TypeError):
        return None
    if expansion_plan is None:
        if num_bundle_rows != origin_batch_size:
            return None
        return tuple(req[b] for b in range(num_bundle_rows))
    sm = expansion_plan.packed_sm_origin or expansion_plan.expanded_to_origin
    if len(sm) < num_bundle_rows:
        return None
    out: list[str] = []
    for j in range(num_bundle_rows):
        o = int(sm[j])
        if o < 0 or o >= len(req):
            return None
        out.append(req[o])
    return tuple(out)


class IntermediatePivotModelProposer(DraftModelProposer):
    """Loads intermediate weights under separate model tag."""

    _INTERMEDIATE_PREFIX = "intermediate_model."

    @override
    def _get_model(self) -> nn.Module:
        from vllm.compilation.backends import set_model_tag

        temp_vllm_config = create_vllm_config_for_draft_model(
            self.vllm_config,
            compile_cache_namespace="intermediate_model",
        )
        with set_model_tag("intermediate_model"):
            return get_model(
                vllm_config=temp_vllm_config,
                prefix="intermediate_model",
            )

    @override
    def load_model(self, target_model: nn.Module) -> None:
        super().load_model(target_model)
        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )
        intermediate_attn_layer_names = {
            name
            for name in all_attn_layers.keys()
            if name.startswith(self._INTERMEDIATE_PREFIX)
        }
        if not intermediate_attn_layer_names:
            sample = sorted(all_attn_layers.keys())[:16]
            raise RuntimeError(
                "Failed to discover intermediate_model attention layers in the "
                "static forward context after loading the intermediate verifier. "
                f"sample_layer_names={sample}"
            )
        self._draft_attn_layer_names = intermediate_attn_layer_names

    @override
    def initialize_attn_backend(
        self,
        kv_cache_config: KVCacheConfig,
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
        super().initialize_attn_backend(kv_cache_config, kernel_block_sizes)
        discovered = {
            layer_name
            for group in self.draft_attn_groups
            for layer_name in group.layer_names
        }
        missing = self._draft_attn_layer_names - discovered
        if missing:
            raise RuntimeError(
                "Intermediate verifier KV/attention backend initialization missed "
                "layers bound in load_model() "
                f"(sample): {sorted(missing)[:32]}"
            )


class PivotDraftModelProposer(PinnedDraftNamespaceDraftModelProposer):
    """Root pivot drafter: explicit ``draft_model.*`` namespace (symmetric to
    :class:`IntermediatePivotModelProposer`).
    """


class PivotTargetDitVerifierProposer(DraftModelProposer):
    """DIT inner verification on the target module (alias `load_model(target)`).

    For ``target_only`` pivot there is no intermediate model; hierarchical
    verification still runs prefix-conditioned forwards that must use target
    hidden sizes and the same attention layers as the main model. We avoid a
    second full target load by binding ``self.model`` to the runner's target.
    """

    @override
    def load_model(self, target_model: nn.Module) -> None:
        target_attn_layer_names = set(
            get_layers_from_vllm_config(
                self.vllm_config,
                AttentionLayerBase,  # type: ignore[type-abstract]
            ).keys()
        )
        self.model = target_model
        # Parent logic subtracts "target" registry from post-draft registry; with no
        # second load that set is empty — use all target attention layers instead.
        self._draft_attn_layer_names = set(target_attn_layer_names)

        if self.supports_mm_inputs:
            try:
                dummy_input_ids = torch.tensor([[1]], device=self.input_ids.device)
                self.model.embed_input_ids(dummy_input_ids, multimodal_embeddings=None)
            except (NotImplementedError, AttributeError, TypeError):
                logger.warning(
                    "Target DIT verifier does not support multimodal embed probe; "
                    "falling back to text-only mode"
                )
                self.supports_mm_inputs = False

        if supports_multimodal(target_model):
            assert hasattr(target_model, "config")
            if self.get_model_name(target_model) in [
                "Qwen2_5_VLForConditionalGeneration",
                "Qwen3VLForConditionalGeneration",
                "Qwen3VLMoeForConditionalGeneration",
                "HunYuanVLForConditionalGeneration",
                "GlmOcrForConditionalGeneration",
                "Qwen3_5ForConditionalGeneration",
                "Qwen3_5MoeForConditionalGeneration",
            ]:
                self.model.config.image_token_index = target_model.config.image_token_id
            elif self.get_model_name(target_model) == "PixtralForConditionalGeneration":
                self.model.config.image_token_index = (
                    target_model.config.vision_config.image_token_id
                )
            else:
                self.model.config.image_token_index = (
                    target_model.config.image_token_index
                )
        # Skip _maybe_share_embeddings / _maybe_share_lm_head: self.model is target.

        if self.parallel_drafting and self.pass_hidden_states_to_model:
            assert self.parallel_drafting_hidden_state_tensor is not None
            self.parallel_drafting_hidden_state_tensor.copy_(
                self.model.combine_hidden_states(
                    self.model.mask_hidden.view(3 * self.hidden_size)
                )
                if self.eagle3_use_aux_hidden_state
                else self.model.mask_hidden.view(self.hidden_size)
            )

        if self.use_local_argmax_reduction:
            if not hasattr(self.model, "get_top_tokens"):
                raise ValueError(
                    "use_local_argmax_reduction is enabled but target verifier "
                    f"{self.model.__class__.__name__} does not implement get_top_tokens()."
                )
            if (
                hasattr(self.model, "draft_id_to_target_id")
                and self.model.draft_id_to_target_id is not None
            ):
                logger.warning(
                    "use_local_argmax_reduction is enabled but model uses "
                    "draft_id_to_target_id vocab remapping; falling back to full logits."
                )
            else:
                logger.info(
                    "Using local argmax reduction for draft token generation "
                    "(communication: O(2*tp_size) vs O(vocab_size))."
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
        self._min_batch_for_expansion = int(spec.pivot_min_batch_for_expansion)
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
            self._draft = PivotDraftModelProposer(
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
        if mode.verification_pipeline in (
            "intermediate_then_target",
            "intermediate_tree_then_target_tree",
        ):
            self._intermediate = IntermediatePivotModelProposer(
                _vllm_intermediate_as_draft(vllm_config, length=self._L),
                device,
                runner,
            )
        else:
            self._intermediate = PivotTargetDitVerifierProposer(
                _vllm_target_as_draft_chunk(vllm_config, length=1),
                device,
                runner,
            )
        # Keep parity with AdaptiveSpechiveProposer contract used by
        # run_hierarchical_verification_rounds().
        self._inter_dit: DraftModelProposer = self._intermediate
        # Staging only: GpuModelRunner must call take_pending_spechive_bundle() then
        # set_pending_hybrid_spec_bundle() once per draft proposal step (single publish).
        self._staged_hybrid_bundle: HybridProposalBundle | None = None
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
        # Frozen after root draft proposal (survives inner HV overwrites of
        # ``last_root_topk_info`` on delegates).
        self._profile_first_draft_topk_info: RootTopKInfo | None = None
        self._profile_first_draft_topk_req_ids: tuple[str, ...] | None = None

    def reset_profile_first_draft_topk(self) -> None:
        self._profile_first_draft_topk_info = None
        self._profile_first_draft_topk_req_ids = None
        for _del in (self._draft, self._eagle_head, self._intermediate):
            if _del is not None and hasattr(_del, "reset_profile_first_draft_topk"):
                _del.reset_profile_first_draft_topk()

    def get_profile_first_draft_topk(
        self,
    ) -> tuple[RootTopKInfo | None, tuple[str, ...] | None]:
        if (
            self._profile_first_draft_topk_info is not None
            and self._profile_first_draft_topk_req_ids is not None
        ):
            return self._profile_first_draft_topk_info, self._profile_first_draft_topk_req_ids
        for _del in (self._draft, self._eagle_head):
            if _del is not None and hasattr(_del, "get_profile_first_draft_topk"):
                topk, snap = _del.get_profile_first_draft_topk()
                if topk is not None and snap is not None:
                    return topk, snap
        return None, None

    def _freeze_profile_first_draft_topk_after_root(
        self, delegate: object, batch_size: int
    ) -> None:
        """Snapshot profile top-k at root-proposal time for unified metadata emission."""
        if self.runner is None or batch_size <= 0:
            return
        prof = getattr(self.runner, "_spec_profiler", None)
        mode = getattr(prof, "mode", None) if prof is not None else None
        if mode is None or not getattr(mode, "metadata_enabled", False):
            return
        get_fn = getattr(delegate, "get_profile_first_draft_topk", None)
        if callable(get_fn):
            topk, snap = get_fn()
            if (
                topk is not None
                and snap is not None
                and len(snap) == batch_size
                and topk.topk_token_ids.shape[0] == batch_size
            ):
                self._profile_first_draft_topk_info = RootTopKInfo(
                    topk_token_ids=topk.topk_token_ids.detach().clone(),
                    topk_probs=topk.topk_probs.detach().clone(),
                )
                self._profile_first_draft_topk_req_ids = snap
                return
        ib = getattr(self.runner, "input_batch", None)
        if ib is None or len(ib.req_ids) < batch_size:
            return
        snap = tuple(str(ib.req_ids[i]) for i in range(batch_size))
        rt = getattr(delegate, "last_root_topk_info", None)
        if rt is None or rt.topk_token_ids.shape[0] != batch_size:
            return
        self._profile_first_draft_topk_info = RootTopKInfo(
            topk_token_ids=rt.topk_token_ids.detach().clone(),
            topk_probs=rt.topk_probs.detach().clone(),
        )
        self._profile_first_draft_topk_req_ids = snap

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

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
        origin_batch_size: int | None = None,
    ) -> None:
        """Warm drafter CUDA graphs for origin-B and packed-P row counts.

        Linear fixed-capacity pivot (no eagle tree, top-k > 1) runs the draft on
        origin batch rows for the root proposal and on packed P rows after top-k
        expansion; both shapes must be registered with the drafter dispatcher.
        """
        if self._draft is not None:
            self._draft.dummy_run(
                num_tokens,
                use_cudagraphs=use_cudagraphs,
                is_graph_capturing=is_graph_capturing,
                slot_mappings=slot_mappings,
            )
        if self._eagle_head is not None:
            self._eagle_head.dummy_run(
                num_tokens,
                use_cudagraphs=use_cudagraphs,
                is_graph_capturing=is_graph_capturing,
                slot_mappings=slot_mappings,
            )
        self._intermediate.dummy_run(
            num_tokens,
            use_cudagraphs=use_cudagraphs,
            is_graph_capturing=is_graph_capturing,
            slot_mappings=slot_mappings,
        )

        sc = self.vllm_config.speculative_config
        assert sc is not None
        if (not sc.pivot_use_eagle_tree) and int(sc.pivot_topk_selection) > 1:
            b = origin_batch_size if origin_batch_size is not None else num_tokens
            packed_num_tokens = sc.pivot_packed_batch_size_for_origin_batch(b)
            if packed_num_tokens != b and packed_num_tokens != num_tokens:
                if self._draft is not None:
                    self._draft.dummy_run(
                        packed_num_tokens,
                        use_cudagraphs=use_cudagraphs,
                        is_graph_capturing=is_graph_capturing,
                        slot_mappings=slot_mappings,
                    )
                if self._eagle_head is not None:
                    self._eagle_head.dummy_run(
                        packed_num_tokens,
                        use_cudagraphs=use_cudagraphs,
                        is_graph_capturing=is_graph_capturing,
                        slot_mappings=slot_mappings,
                    )
                self._intermediate.dummy_run(
                    packed_num_tokens,
                    use_cudagraphs=use_cudagraphs,
                    is_graph_capturing=is_graph_capturing,
                    slot_mappings=slot_mappings,
                )

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
        bundle = self._staged_hybrid_bundle
        self._staged_hybrid_bundle = None
        return bundle

    def discard_pending_hierarchical_verification_state(self) -> None:
        self._staged_hybrid_bundle = None
        self._active_pivot_expansion_plan = None
        self._pending_tree_plan = None
        self._is_waiting_for_target_collapse = False
        self._pending_pivot_bundle_metadata = None

    def supports_staged_eagle_fastpath(self) -> bool:
        if not self._pivot_spechive:
            return False
        if self._pivot_mode.proposal_engine != "eagle3_head":
            return False
        sd = self._staged_delegates
        return sd is not None and sd.hidden_state_provider is not None

    def bootstrap_intermediate_round_state(
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
        provider = self._staged_delegates.hidden_state_provider
        assert provider is not None
        return provider.bootstrap_from_current_prefix(
            base_target_token_ids=base_target_token_ids,
            base_target_positions=base_target_positions,
            base_target_hidden_states=base_target_hidden_states,
            base_next_token_ids=base_next_token_ids,
            base_common_attn_metadata=base_common_attn_metadata,
            base_num_rejected_tokens_gpu=base_num_rejected_tokens_gpu,
            prefix_rows=prefix_rows,
        )

    def refresh_intermediate_round_state(
        self,
        *,
        base_target_token_ids: torch.Tensor,
        base_target_positions: torch.Tensor,
        base_target_hidden_states: torch.Tensor,
        base_next_token_ids: torch.Tensor,
        base_common_attn_metadata: CommonAttentionMetadata,
        base_num_rejected_tokens_gpu: torch.Tensor | None,
        prefix_rows: list[list[int]],
        old_state: IntermediateRoundState | None,
    ) -> IntermediateRoundState:
        # Same contract as adaptive: re-bootstrap hidden bundle for the new
        # prefix; ``old_state`` reserved for future incremental refresh.
        del old_state
        return self.bootstrap_intermediate_round_state(
            base_target_token_ids=base_target_token_ids,
            base_target_positions=base_target_positions,
            base_target_hidden_states=base_target_hidden_states,
            base_next_token_ids=base_next_token_ids,
            base_common_attn_metadata=base_common_attn_metadata,
            base_num_rejected_tokens_gpu=base_num_rejected_tokens_gpu,
            prefix_rows=prefix_rows,
        )

    def propose_chunk_from_intermediate_state(
        self,
        *,
        inter_state: IntermediateRoundState,
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
        from vllm.v1.spec_decode.staged_eagle import EagleHeadProposalEngine

        assert inter_state.hidden_bundle is not None
        eng = self._staged_delegates.proposal_engine
        assert isinstance(eng, EagleHeadProposalEngine)
        rows, probs = eng.propose_from_hidden_states(
            base_target_token_ids=base_target_token_ids,
            base_target_positions=base_target_positions,
            hidden_bundle=inter_state.hidden_bundle,
            base_next_token_ids=base_next_token_ids,
            base_common_attn_metadata=base_common_attn_metadata,
            base_num_rejected_tokens_gpu=base_num_rejected_tokens_gpu,
            prefix_rows=prefix_rows,
            chunk_len=chunk_len,
            sampling_metadata=sampling_metadata,
            use_draft_probs=use_draft_probs,
        )
        return DitRoundProposal(tokens=rows, probs=probs)

    def verify_chunk_with_intermediate_state(
        self,
        *,
        inter_state: IntermediateRoundState,
        base_target_token_ids: torch.Tensor,
        base_target_positions: torch.Tensor,
        base_target_hidden_states: torch.Tensor,
        base_next_token_ids: torch.Tensor,
        base_common_attn_metadata: CommonAttentionMetadata,
        base_num_rejected_tokens_gpu: torch.Tensor | None,
        prefix_rows: list[list[int]],
        candidate_tokens: torch.Tensor,
        mirror_kv_common_attn_metadata: CommonAttentionMetadata | None = None,
    ) -> DitRoundVerification:
        return self.verify_chunk_with_inter_verifier(
            base_target_token_ids=base_target_token_ids,
            base_target_positions=base_target_positions,
            base_target_hidden_states=base_target_hidden_states,
            base_next_token_ids=base_next_token_ids,
            base_common_attn_metadata=base_common_attn_metadata,
            base_num_rejected_tokens_gpu=base_num_rejected_tokens_gpu,
            prefix_rows=prefix_rows,
            candidate_tokens=candidate_tokens,
            reuse_intermediate_state=inter_state,
            mirror_kv_common_attn_metadata=mirror_kv_common_attn_metadata,
        )

    def _select_low_confidence_indices(self, pivot_probs: torch.Tensor) -> list[int]:
        batch_size = int(pivot_probs.shape[0])
        if batch_size <= 0:
            return []
        num_expand = int(math.ceil(batch_size * self._expansion_pct))
        num_expand = min(max(num_expand, 0), batch_size)
        if num_expand == 0:
            return []
        top1_prob = pivot_probs[:, 0, :].amax(dim=-1)
        return self._select_low_confidence_indices_from_top1_prob(top1_prob)

    def _select_low_confidence_indices_from_top1_prob(
        self, top1_prob: torch.Tensor
    ) -> list[int]:
        batch_size = int(top1_prob.shape[0])
        if batch_size <= 0:
            return []
        num_expand = int(math.ceil(batch_size * self._expansion_pct))
        num_expand = min(max(num_expand, 0), batch_size)
        if num_expand == 0:
            return []
        chosen = torch.topk(top1_prob, k=num_expand, largest=False).indices
        return [int(i) for i in chosen.tolist()]

    def _select_low_confidence_origin_rows(self, pivot_probs: torch.Tensor) -> list[int]:
        return self._select_low_confidence_indices(pivot_probs)

    def _build_expanded_root_families(
        self,
        *,
        expansion_plan: PivotExpansionPlan,
    ) -> list[PivotTreeFamily]:
        """One ``PivotTreeFamily`` per packed proposal row ``j`` in ``0..P-1``."""
        families: list[PivotTreeFamily] = []
        row_meta: dict[int, tuple[int, int]] = {}
        for fam in expansion_plan.families:
            for local_idx, row_idx in enumerate(fam.expanded_rows):
                row_meta[int(row_idx)] = (
                    int(fam.candidate_ranks[local_idx]),
                    int(fam.first_token_ids[local_idx]),
                )
        p = int(expansion_plan.expanded_batch_size)
        eto = expansion_plan.expanded_to_origin
        if len(eto) != p:
            return families
        for j in range(p):
            origin_row = int(eto[j])
            root_rank, root_token_id = row_meta.get(j, (0, 0))
            families.append(
                PivotTreeFamily(
                    origin_row=origin_row,
                    family_id=int(j),
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
        b_origin = (
            int(expansion_plan.origin_batch_size)
            if expansion_plan.origin_batch_size > 0
            else max(expansion_plan.expanded_to_origin) + 1
        )
        flat_plan = build_family_flatten_order(
            families=families,
            template=template,
            origin_batch_size=b_origin,
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
        pivot_probs: torch.Tensor | None,
        enable_topk_expansion: bool,
        root_topk_info: RootTopKInfo | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, PivotExpansionPlan | None]:
        """Build packed pivot rows (fixed P) or legacy variable rows (eagle tree)."""
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

        if pivot_probs is None:
            if root_topk_info is None:
                ep = initial_pivots.to(torch.int32)
                return ep, None, None
            if int(root_topk_info.topk_token_ids.shape[0]) != batch_size:
                ep = initial_pivots.to(torch.int32)
                return ep, None, None
            return self._build_pivot_expansion_plan_fixed_capacity_from_root_topk(
                initial_pivots=initial_pivots,
                root_topk=root_topk_info,
            )

        # Eagle tree uses the same fixed P layout as linear pivot (PR2) so tree
        # families, bundle rows, and verifier metadata stay aligned at P.
        if self._pivot_use_eagle_tree:
            return self._build_pivot_expansion_plan_fixed_capacity(
                initial_pivots=initial_pivots,
                pivot_probs=pivot_probs,
            )

        return self._build_pivot_expansion_plan_fixed_capacity(
            initial_pivots=initial_pivots,
            pivot_probs=pivot_probs,
        )

    def _build_pivot_expansion_plan_legacy_variable(
        self,
        *,
        initial_pivots: torch.Tensor,
        pivot_probs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, PivotExpansionPlan | None]:
        """Variable B' expansion (pre-PR2). Retained for reference; eagle tree uses fixed P."""
        batch_size = int(initial_pivots.shape[0])
        selected_set = set(self._select_low_confidence_indices(pivot_probs))
        if not selected_set:
            ep = initial_pivots.to(torch.int32)
            return ep, pivot_probs[:, :1, :].clone(), None

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
        p_len = len(expanded_to_origin)
        seen_o: set[int] = set()
        is_base_flags: list[bool] = []
        for o in expanded_to_origin:
            is_base_flags.append(o not in seen_o)
            seen_o.add(o)
        origin_to_base_row = [-1] * batch_size
        for i, o in enumerate(expanded_to_origin):
            if origin_to_base_row[o] < 0:
                origin_to_base_row[o] = i
        eto = list(expanded_to_origin)
        plan = PivotExpansionPlan(
            expanded_to_origin=eto,
            families=families,
            expanded_batch_size=p_len,
            origin_batch_size=batch_size,
            packed_batch_size=p_len,
            packed_to_origin=list(eto),
            packed_sm_origin=list(eto),
            packed_row_is_active=[True] * p_len,
            packed_row_is_base=is_base_flags,
            packed_row_family_rank=[0] * p_len,
            origin_to_base_row=origin_to_base_row,
            origin_to_family_rows=[[] for _ in range(batch_size)],
            uses_fixed_capacity_packing=False,
        )
        idx = torch.tensor(expanded_to_origin, device=pivot_probs.device, dtype=torch.long)
        expanded_probs = pivot_probs[idx, :1, :].clone()
        return expanded_pivots, expanded_probs, plan

    def _build_pivot_expansion_plan_fixed_capacity(
        self,
        *,
        initial_pivots: torch.Tensor,
        pivot_probs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, PivotExpansionPlan | None]:
        """Fixed P = B + ceil(B*pct)*(K-1); Option 2 — always packed when top-k on."""
        B = int(initial_pivots.shape[0])
        K = min(self._topk_selection, int(pivot_probs.shape[-1]))
        num_expand = int(math.ceil(B * self._expansion_pct))
        p_extra = max(0, K - 1)
        P = B + num_expand * p_extra
        base_piv = initial_pivots.to(torch.int32).view(-1)

        selected_set = set(self._select_low_confidence_indices(pivot_probs))
        selected_sorted = sorted(selected_set)[:num_expand]
        if _spechive_debug_enabled():
            logger.info(
                "PIVOT_DEBUG plan: B=%d K=%d pct=%.3f num_expand=%d P=%d p_extra=%d selected=%s",
                B,
                K,
                self._expansion_pct,
                num_expand,
                P,
                p_extra,
                selected_sorted,
            )

        packed_to_origin = [-1] * P
        packed_sm_origin = [0] * P
        is_active = [False] * P
        is_base = [False] * P
        fam_rank = [-1] * P
        for o in range(B):
            packed_to_origin[o] = o
            packed_sm_origin[o] = o
            is_active[o] = True
            is_base[o] = True
            fam_rank[o] = 0

        pivot_vals = [0] * P
        for o in range(B):
            pivot_vals[o] = int(base_piv[o].item())
        families: list[PivotExpansionFamily] = []
        origin_to_family_rows: list[list[int]] = [[] for _ in range(B)]

        for b in range(num_expand):
            owner = selected_sorted[b] if b < len(selected_sorted) else (b % B)
            for r in range(p_extra):
                j = B + b * p_extra + r
                packed_sm_origin[j] = owner
                if b < len(selected_sorted):
                    packed_to_origin[j] = owner
                    is_active[j] = True
                    fam_rank[j] = r + 1
                else:
                    packed_to_origin[j] = -1
                    is_active[j] = False
                    fam_rank[j] = -1
                    pivot_vals[j] = int(base_piv[owner].item())
            if b < len(selected_sorted):
                o = int(owner)
                cand_ids = torch.topk(pivot_probs[o, 0], k=K, largest=True).indices.tolist()
                cand_ids = [int(t) for t in cand_ids]
                cand_probs = [float(pivot_probs[o, 0, t].item()) for t in cand_ids]
                base_row = o
                expanded_rows = [base_row]
                for r in range(p_extra):
                    jj = B + b * p_extra + r
                    expanded_rows.append(jj)
                    pivot_vals[jj] = cand_ids[r + 1]
                families.append(
                    PivotExpansionFamily(
                        origin_row=o,
                        expanded_rows=expanded_rows,
                        candidate_ranks=list(range(len(cand_ids))),
                        first_token_ids=cand_ids,
                        first_token_probs=cand_probs,
                    )
                )
                for r in range(p_extra):
                    origin_to_family_rows[o].append(B + b * p_extra + r)

        device = initial_pivots.device
        expanded_pivots = torch.tensor(pivot_vals, device=device, dtype=torch.int32).view(
            P, 1
        )
        sm_idx = torch.tensor(packed_sm_origin, device=pivot_probs.device, dtype=torch.long)
        expanded_probs = pivot_probs[sm_idx, :1, :].clone()
        expanded_to_origin = list(packed_sm_origin)
        plan = PivotExpansionPlan(
            expanded_to_origin=expanded_to_origin,
            families=families,
            expanded_batch_size=P,
            origin_batch_size=B,
            packed_batch_size=P,
            packed_to_origin=list(packed_to_origin),
            packed_sm_origin=list(packed_sm_origin),
            packed_row_is_active=list(is_active),
            packed_row_is_base=list(is_base),
            packed_row_family_rank=list(fam_rank),
            origin_to_base_row=list(range(B)),
            origin_to_family_rows=origin_to_family_rows,
            uses_fixed_capacity_packing=True,
        )
        return expanded_pivots, expanded_probs, plan

    def _build_pivot_expansion_plan_fixed_capacity_from_root_topk(
        self,
        *,
        initial_pivots: torch.Tensor,
        root_topk: RootTopKInfo,
    ) -> tuple[torch.Tensor, torch.Tensor | None, PivotExpansionPlan | None]:
        """Same fixed P layout as ``_build_pivot_expansion_plan_fixed_capacity`` using top-k only."""
        B = int(initial_pivots.shape[0])
        k_width = int(root_topk.topk_token_ids.shape[1])
        K = min(self._topk_selection, k_width)
        num_expand = int(math.ceil(B * self._expansion_pct))
        p_extra = max(0, K - 1)
        P = B + num_expand * p_extra
        base_piv = initial_pivots.to(torch.int32).view(-1)

        top1_prob = root_topk.topk_probs[:, 0]
        selected_set = set(self._select_low_confidence_indices_from_top1_prob(top1_prob))
        selected_sorted = sorted(selected_set)[:num_expand]
        if _spechive_debug_enabled():
            logger.info(
                "PIVOT_DEBUG plan: B=%d K=%d pct=%.3f num_expand=%d P=%d p_extra=%d selected=%s "
                "(root_topk)",
                B,
                K,
                self._expansion_pct,
                num_expand,
                P,
                p_extra,
                selected_sorted,
            )

        packed_to_origin = [-1] * P
        packed_sm_origin = [0] * P
        is_active = [False] * P
        is_base = [False] * P
        fam_rank = [-1] * P
        for o in range(B):
            packed_to_origin[o] = o
            packed_sm_origin[o] = o
            is_active[o] = True
            is_base[o] = True
            fam_rank[o] = 0

        pivot_vals = [0] * P
        for o in range(B):
            pivot_vals[o] = int(base_piv[o].item())
        families: list[PivotExpansionFamily] = []
        origin_to_family_rows: list[list[int]] = [[] for _ in range(B)]

        for b in range(num_expand):
            owner = selected_sorted[b] if b < len(selected_sorted) else (b % B)
            for r in range(p_extra):
                j = B + b * p_extra + r
                packed_sm_origin[j] = owner
                if b < len(selected_sorted):
                    packed_to_origin[j] = owner
                    is_active[j] = True
                    fam_rank[j] = r + 1
                else:
                    packed_to_origin[j] = -1
                    is_active[j] = False
                    fam_rank[j] = -1
                    pivot_vals[j] = int(base_piv[owner].item())
            if b < len(selected_sorted):
                o = int(owner)
                cand_ids = [
                    int(root_topk.topk_token_ids[o, j].item()) for j in range(K)
                ]
                cand_probs = [
                    float(root_topk.topk_probs[o, j].item()) for j in range(K)
                ]
                base_row = o
                expanded_rows = [base_row]
                for r in range(p_extra):
                    jj = B + b * p_extra + r
                    expanded_rows.append(jj)
                    pivot_vals[jj] = cand_ids[r + 1]
                families.append(
                    PivotExpansionFamily(
                        origin_row=o,
                        expanded_rows=expanded_rows,
                        candidate_ranks=list(range(len(cand_ids))),
                        first_token_ids=cand_ids,
                        first_token_probs=cand_probs,
                    )
                )
                for r in range(p_extra):
                    origin_to_family_rows[o].append(B + b * p_extra + r)

        device = initial_pivots.device
        expanded_pivots = torch.tensor(pivot_vals, device=device, dtype=torch.int32).view(
            P, 1
        )
        expanded_to_origin = list(packed_sm_origin)
        plan = PivotExpansionPlan(
            expanded_to_origin=expanded_to_origin,
            families=families,
            expanded_batch_size=P,
            origin_batch_size=B,
            packed_batch_size=P,
            packed_to_origin=list(packed_to_origin),
            packed_sm_origin=list(packed_sm_origin),
            packed_row_is_active=list(is_active),
            packed_row_is_base=list(is_base),
            packed_row_family_rank=list(fam_rank),
            origin_to_base_row=list(range(B)),
            origin_to_family_rows=origin_to_family_rows,
            uses_fixed_capacity_packing=True,
        )
        return expanded_pivots, None, plan

    def _pivot_probs_sparse_from_root_topk_for_plan(
        self,
        plan: PivotExpansionPlan,
        root_topk: RootTopKInfo,
        vocab_size: int,
    ) -> torch.Tensor:
        """Build ``[P, 1, V]`` root-step probs by scattering top-k rows per origin.

        Used when expansion came from ``RootTopKInfo`` only (no full delegate
        ``last_draft_probs_flat``) so hybrid bundles still carry per-row q(·) for
        rejection / non-greedy paths after ``clear_draft_probs()`` on the delegate.
        """
        device = root_topk.topk_token_ids.device
        dtype = torch.float32
        p_len = int(plan.expanded_batch_size)
        sm = plan.packed_sm_origin
        if sm is None:
            sm = plan.expanded_to_origin
        b_origin = int(root_topk.topk_token_ids.shape[0])
        k_w = int(root_topk.topk_token_ids.shape[1])
        out = torch.zeros(p_len, 1, vocab_size, device=device, dtype=dtype)
        for j in range(p_len):
            o = int(sm[j])
            if o < 0 or o >= b_origin:
                continue
            tid = root_topk.topk_token_ids[o, :k_w].long().clamp(0, vocab_size - 1)
            tpr = root_topk.topk_probs[o, :k_w].to(dtype=dtype)
            out[j, 0].scatter_(0, tid, tpr)
        return out

    @staticmethod
    def _expand_prefix_rows_for_plan(
        base_prefix_rows: list[list[int]],
        plan: PivotExpansionPlan,
    ) -> list[list[int]]:
        sm = plan.packed_sm_origin
        if sm is None:
            sm = plan.expanded_to_origin
        return [list(base_prefix_rows[o]) for o in sm]

    def _expand_sampling_metadata_for_plan(
        self,
        sampling_metadata: SamplingMetadata,
        plan: PivotExpansionPlan,
        expanded_prefix_rows: list[list[int]],
    ) -> SamplingMetadata:
        sm = plan.packed_sm_origin
        if sm is None:
            sm = plan.expanded_to_origin
        return slice_sampling_metadata_for_subbatch(
            sampling_metadata,
            sm,
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
        reuse_intermediate_state: IntermediateRoundState | None = None,
    ) -> tuple[
        torch.Tensor,
        tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, CommonAttentionMetadata
        ]
        | None,
    ]:
        if self._pivot_mode.proposal_engine != "eagle3_head":
            return base_target_hidden_states, None
        if self._pivot_mode.hidden_state_source == "target":
            return base_target_hidden_states, None
        if (
            reuse_intermediate_state is not None
            and reuse_intermediate_state.hidden_bundle is not None
        ):
            hb = reuse_intermediate_state.hidden_bundle
            return hb.hidden_states, hb.prefix_prefab
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
        bundle = round_state.hidden_bundle
        return bundle.hidden_states, bundle.prefix_prefab

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

    @staticmethod
    def _build_expansion_gather_indices(
        plan: PivotExpansionPlan,
        base_query_start_loc: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build GPU index tensors for expanded batch P from origin batch B.

        Returns ``(flat_token_gather_idx, row_gather_idx, new_query_start_loc)``
        where:
        - ``row_gather_idx``  shape ``[P]``: maps each expanded row to its origin
          row index in ``[0, B)``.
        - ``flat_token_gather_idx`` shape ``[total_expanded_tokens]``: maps each
          flat token position in the expanded layout back to a position in the
          original flat token layout.
        - ``new_query_start_loc`` shape ``[P+1]``: cumulative token offsets for
          the expanded batch.
        """
        p_len = int(plan.expanded_batch_size)
        sm = plan.packed_sm_origin
        if sm is None:
            sm = plan.expanded_to_origin
        assert len(sm) == p_len

        device = base_query_start_loc.device
        row_gather_idx = torch.tensor(sm, dtype=torch.long, device=device)

        origin_query_lens = (
            base_query_start_loc[1:] - base_query_start_loc[:-1]
        )
        expanded_query_lens = origin_query_lens[row_gather_idx]

        new_qsl = torch.zeros(
            p_len + 1, dtype=base_query_start_loc.dtype, device=device
        )
        torch.cumsum(expanded_query_lens, dim=0, out=new_qsl[1:])
        total_tokens = int(new_qsl[p_len].item())

        origin_starts = base_query_start_loc[row_gather_idx]

        # Vectorised: for each expanded token, compute its source index in
        # the original flat token layout.  token_row_id maps each expanded
        # token to its expanded-row index; origin offset then gives the
        # source start for that row.
        token_row_id = torch.repeat_interleave(
            torch.arange(p_len, device=device),
            expanded_query_lens,
        )
        flat_token_gather_idx = (
            torch.arange(total_tokens, device=device)
            - new_qsl[token_row_id]
            + origin_starts[token_row_id]
        )

        return flat_token_gather_idx, row_gather_idx, new_qsl

    def _expand_tail_proposer_frontier_for_plan(
        self,
        *,
        plan: PivotExpansionPlan,
        base_target_token_ids: torch.Tensor,
        base_target_positions: torch.Tensor,
        base_target_hidden_states: torch.Tensor,
        base_next_token_ids: torch.Tensor,
        base_common_attn_metadata: CommonAttentionMetadata,
        base_num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        CommonAttentionMetadata,
        torch.Tensor | None,
    ]:
        """Repeat origin frontier tensors/metadata once per packed pivot row (length P).

        Uses index-based gather instead of per-row torch.cat to avoid
        materializing intermediate expanded tensors.
        """
        import time as _time

        _t0 = _time.perf_counter() if _spechive_debug_enabled() else 0.0

        cad = base_common_attn_metadata
        _prof = (
            getattr(self.runner, "_spec_profiler", None)
            if self.runner is not None
            else None
        )
        base_qsl_tail: int | None = None
        base_seq_sum: int | None = None
        if _spechive_debug_enabled():
            base_qsl_tail = int(cad.query_start_loc[-1].item())
            base_seq_sum = int(cad.seq_lens.sum().item())
        qsl = cad.query_start_loc
        p_len = int(plan.expanded_batch_size)

        if _prof is not None:
            _prof.start_stage("expand_frontier_gather")
        flat_idx, row_idx, new_qsl = self._build_expansion_gather_indices(
            plan, qsl
        )
        total_tokens = int(new_qsl[p_len].item())

        out_tokens = base_target_token_ids.index_select(0, flat_idx)
        out_hidden = base_target_hidden_states.index_select(0, flat_idx)
        out_slot = cad.slot_mapping.index_select(0, flat_idx)

        positions_2d = base_target_positions.dim() > 1
        if positions_2d:
            out_positions = base_target_positions.index_select(1, flat_idx)
        else:
            out_positions = base_target_positions.index_select(0, flat_idx)

        out_next = base_next_token_ids.index_select(0, row_idx).to(torch.int32)
        out_seq_lens = cad.seq_lens.index_select(0, row_idx)
        out_block = cad.block_table_tensor.index_select(0, row_idx)

        out_rej: torch.Tensor | None = None
        if base_num_rejected_tokens_gpu is not None:
            out_rej = base_num_rejected_tokens_gpu.index_select(0, row_idx)

        out_dcp: torch.Tensor | None = None
        if cad.dcp_local_seq_lens is not None:
            out_dcp = cad.dcp_local_seq_lens.index_select(0, row_idx)

        out_lip: torch.Tensor | None = None
        if cad.logits_indices_padded is not None:
            out_lip = cad.logits_indices_padded.index_select(0, row_idx)
        if _prof is not None:
            _prof.end_stage("expand_frontier_gather")

        query_lens = new_qsl[1:] - new_qsl[:-1]
        max_q = int(query_lens.max().item()) if p_len > 0 else 0

        out_qsl_cpu = None
        out_seq_lens_cpu = None
        if _prof is not None:
            _prof.start_stage("expand_metadata_cpu")
        if cad.query_start_loc_cpu is not None and cad._seq_lens_cpu is not None:
            sm = (
                plan.packed_sm_origin
                if plan.packed_sm_origin is not None
                else plan.expanded_to_origin
            )
            row_idx_cpu = torch.tensor(sm, dtype=torch.long)
            origin_qsl_cpu = cad.query_start_loc_cpu
            origin_query_lens_cpu = origin_qsl_cpu[1:] - origin_qsl_cpu[:-1]
            expanded_query_lens_cpu = origin_query_lens_cpu.index_select(0, row_idx_cpu)
            out_qsl_cpu = torch.empty(p_len + 1, dtype=origin_qsl_cpu.dtype)
            out_qsl_cpu[0] = 0
            if p_len > 0:
                torch.cumsum(expanded_query_lens_cpu, dim=0, out=out_qsl_cpu[1:])
            out_seq_lens_cpu = cad._seq_lens_cpu.index_select(0, row_idx_cpu)
        if _prof is not None:
            _prof.end_stage("expand_metadata_cpu")

        if _spechive_debug_enabled():
            _elapsed_us = (_time.perf_counter() - _t0) * 1e6
            assert base_qsl_tail is not None and base_seq_sum is not None
            assert int(cad.query_start_loc[-1].item()) == base_qsl_tail
            assert int(cad.seq_lens.sum().item()) == base_seq_sum
            logger.info(
                "PIVOT_DEBUG expand_frontier: P=%d total_tokens=%d "
                "elapsed_us=%.1f (index-gather path, no CAD clone)",
                p_len,
                total_tokens,
                _elapsed_us,
            )

        out_cad = cad.replace(
            query_start_loc=new_qsl,
            query_start_loc_cpu=out_qsl_cpu,
            seq_lens=out_seq_lens,
            block_table_tensor=out_block,
            slot_mapping=out_slot,
            num_reqs=p_len,
            num_actual_tokens=total_tokens,
            max_query_len=max_q,
            max_seq_len=int(out_seq_lens.max().item()) if p_len > 0 else 0,
            dcp_local_seq_lens=out_dcp,
            dcp_local_seq_lens_cpu=None,
            logits_indices_padded=out_lip,
            num_logits_indices=None,
            _seq_lens_cpu=out_seq_lens_cpu,
            _num_computed_tokens_cpu=None,
            _num_computed_tokens_cache=None,
        )
        return (
            out_tokens,
            out_positions,
            out_hidden,
            out_next,
            out_cad,
            out_rej,
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
        expansion_plan: PivotExpansionPlan | None = None,
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
            tail_cad = base_common_attn_metadata
            tail_tok = base_target_token_ids
            tail_pos = base_target_positions
            tail_next = base_next_token_ids
            tail_rej = base_num_rejected_tokens_gpu
            if expansion_plan is not None:
                (
                    tail_tok,
                    tail_pos,
                    _tail_hid_base,
                    tail_next,
                    tail_cad,
                    tail_rej,
                ) = self._expand_tail_proposer_frontier_for_plan(
                    plan=expansion_plan,
                    base_target_token_ids=base_target_token_ids,
                    base_target_positions=base_target_positions,
                    base_target_hidden_states=base_target_hidden_states,
                    base_next_token_ids=base_next_token_ids,
                    base_common_attn_metadata=base_common_attn_metadata,
                    base_num_rejected_tokens_gpu=base_num_rejected_tokens_gpu,
                )
                assert int(tail_cad.batch_size()) == len(full_prefix_rows) == batch_size
                if _spechive_debug_enabled():
                    logger.info(
                        "PIVOT_DEBUG tail_contract: prefix_rows=%d cad_rows=%d next_rows=%d",
                        len(full_prefix_rows),
                        int(tail_cad.batch_size()),
                        int(tail_next.shape[0]),
                    )
                proposal_hidden_states, tail_prefix_prefab = (
                    self._resolve_hidden_states_for_proposal(
                        base_target_token_ids=tail_tok,
                        base_target_positions=tail_pos,
                        base_target_hidden_states=_tail_hid_base,
                        base_next_token_ids=tail_next,
                        base_common_attn_metadata=tail_cad,
                        base_num_rejected_tokens_gpu=tail_rej,
                        prefix_rows=full_prefix_rows,
                    )
                )
            else:
                proposal_hidden_states, tail_prefix_prefab = (
                    self._resolve_hidden_states_for_proposal(
                        base_target_token_ids=base_target_token_ids,
                        base_target_positions=base_target_positions,
                        base_target_hidden_states=base_target_hidden_states,
                        base_next_token_ids=base_next_token_ids,
                        base_common_attn_metadata=base_common_attn_metadata,
                        base_num_rejected_tokens_gpu=base_num_rejected_tokens_gpu,
                        prefix_rows=full_prefix_rows,
                    )
                )
            tail_rows, tail_probs = _propose_chunk_from_prefix(
                self._main_delegate(),
                cad=tail_cad,
                target_token_ids=tail_tok,
                target_positions=tail_pos,
                target_hidden_states=proposal_hidden_states,
                next_token_ids=tail_next,
                num_rejected_tokens_gpu=tail_rej,
                prefix_rows=full_prefix_rows,
                chunk_len=tail_len,
                sampling_metadata=tail_sm,
                use_draft_probs=use_draft_probs,
                prefix_prefab=tail_prefix_prefab,
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
        reuse_intermediate_state: IntermediateRoundState | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, PivotExpansionPlan | None]:
        proposal_hidden_states, pivot_prefix_prefab = (
            self._resolve_hidden_states_for_proposal(
                base_target_token_ids=base_target_token_ids,
                base_target_positions=base_target_positions,
                base_target_hidden_states=base_target_hidden_states,
                base_next_token_ids=base_next_token_ids,
                base_common_attn_metadata=base_common_attn_metadata,
                base_num_rejected_tokens_gpu=base_num_rejected_tokens_gpu,
                prefix_rows=base_prefix_rows,
                reuse_intermediate_state=reuse_intermediate_state,
            )
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
            use_draft_probs=use_draft_probs,
            prefix_prefab=pivot_prefix_prefab,
        )
        pivots = pivots.to(torch.int32)[:, :1]
        batch_size = int(pivots.shape[0])
        delegate = self._main_delegate()
        root_topk = getattr(delegate, "last_root_topk_info", None)
        self._freeze_profile_first_draft_topk_after_root(delegate, batch_size)
        chosen_pivots, pivot_probs_rows, expansion_plan = self._build_pivot_expansion_plan(
            initial_pivots=pivots,
            pivot_probs=pivot_probs,
            enable_topk_expansion=enable_topk_expansion,
            root_topk_info=(
                root_topk
                if pivot_probs is None and root_topk is not None
                else None
            ),
        )
        if (
            expansion_plan is not None
            and pivot_probs_rows is None
            and root_topk is not None
        ):
            pivot_probs_rows = self._pivot_probs_sparse_from_root_topk_for_plan(
                expansion_plan,
                root_topk,
                self.vllm_config.model_config.get_vocab_size(),
            )
        if _spechive_debug_enabled():
            logger.info(
                "PIVOT_DEBUG root: verification_rows=%d has_plan=%s expanded_batch=%s "
                "used_root_topk=%s used_full_probs=%s",
                int(pivots.shape[0]),
                expansion_plan is not None,
                expansion_plan.expanded_batch_size if expansion_plan is not None else None,
                pivot_probs is None and root_topk is not None,
                pivot_probs is not None,
            )
        if expansion_plan is not None:
            _ec_prof = (
                getattr(self.runner, "_spec_profiler", None)
                if self.runner is not None
                else None
            )
            _pctx = (
                getattr(self.runner, "_current_profile_ctx", None)
                if self.runner is not None
                else None
            )
            if _ec_prof is not None and _pctx is not None:
                from vllm.v1.spec_decode.profiler_types import (  # noqa: E402
                    SpecDecodeFamilyMetadataRecord,
                )

                _exp_pct = (
                    ((int(expansion_plan.expanded_batch_size) - int(batch_size))
                     / int(batch_size) * 100.0)
                    if int(batch_size) > 0
                    else 0.0
                )
                for _fi, _fam in enumerate(expansion_plan.families):
                    _origin_b = int(getattr(_fam, "origin_row", -1))
                    _origin_rid = (
                        self.runner.input_batch.req_ids[_origin_b]
                        if 0 <= _origin_b < len(self.runner.input_batch.req_ids)
                        else ""
                    )
                    _ec_prof.emit_family_metadata(
                        SpecDecodeFamilyMetadataRecord(
                            step_id=getattr(_pctx, "step_id", 0),
                            family_id=f"{_origin_rid}:f{_fi}",
                            origin_req_id=_origin_rid,
                            family_stage="expand",
                            family_width_before=1,
                            family_width_after=len(getattr(_fam, "expanded_rows", [])),
                            topk_k=len(getattr(_fam, "candidate_ranks", [])),
                            expansion_pct=_exp_pct,
                        )
                    )
        if expansion_plan is not None:
            self._active_pivot_expansion_plan = expansion_plan
            if self._pivot_spechive:
                self._is_waiting_for_target_collapse = True
        elif not self._pivot_spechive:
            self._active_pivot_expansion_plan = None

        _ec_prof = (
            getattr(self.runner, "_spec_profiler", None)
            if self.runner is not None
            else None
        )
        if expansion_plan is not None:
            if _ec_prof is not None:
                _ec_prof.start_stage("expand_collapse", invocation_idx=0)
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
            expansion_plan=expansion_plan,
        )
        if expansion_plan is not None and expansion_plan.packed_row_is_active is not None:
            for j in range(int(out.shape[0])):
                if not expansion_plan.packed_row_is_active[j]:
                    out[j].fill_(PLACEHOLDER_TOKEN_ID)
        if expansion_plan is not None and _ec_prof is not None:
            _ec_prof.end_stage("expand_collapse", invocation_idx=0)
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
        proposal_hidden_states, tree_pivot_prefab = (
            self._resolve_hidden_states_for_proposal(
                base_target_token_ids=target_token_ids,
                base_target_positions=target_positions,
                base_target_hidden_states=target_hidden_states,
                base_next_token_ids=next_token_ids,
                base_common_attn_metadata=common_attn_metadata,
                base_num_rejected_tokens_gpu=num_rejected_tokens_gpu,
                prefix_rows=base_prefix_rows,
            )
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
            prefix_prefab=tree_pivot_prefab,
        )
        pivots = pivots.to(torch.int32)[:, :1]
        rt = getattr(self._eagle_head, "last_root_topk_info", None)
        self._freeze_profile_first_draft_topk_after_root(
            self._eagle_head, batch_size)
        chosen_pivots, _, expansion_plan = self._build_pivot_expansion_plan(
            initial_pivots=pivots,
            pivot_probs=pivot_probs,
            enable_topk_expansion=True,
            root_topk_info=rt if pivot_probs is None and rt is not None else None,
        )
        if expansion_plan is not None:
            self._active_pivot_expansion_plan = expansion_plan
            if self._pivot_spechive:
                self._is_waiting_for_target_collapse = True
            rows_per_family: list[torch.Tensor] = []
            sm_eagle = expansion_plan.packed_sm_origin or expansion_plan.expanded_to_origin
            for fam_row, origin_row in enumerate(sm_eagle):
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
        elif (
            use_draft_probs
            and expansion_plan is not None
            and expansion_plan.uses_fixed_capacity_packing
        ):
            # Fixed P: one tree proposal row per packed index; eagle stores flat
            # probs aligned to ``rows.reshape(-1)`` when available.
            probs_flat = getattr(self._eagle_head, "last_draft_probs_flat", None)
            if probs_flat is not None and int(probs_flat.shape[0]) != int(
                rows.numel()
            ):
                probs_flat = None
        elif expansion_plan is not None:
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
        reuse_intermediate_state: IntermediateRoundState | None = None,
    ) -> DitRoundProposal:
        origin_bs = int(base_common_attn_metadata.batch_size())
        enable_topk = (
            self._topk_selection > 1
            and origin_bs >= self._min_batch_for_expansion
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
            reuse_intermediate_state=reuse_intermediate_state,
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
            # Legacy variable expansion: shared origin q(.) is unsafe for rejection.
            # Fixed-capacity path: per-row probs from compose / eagle when aligned.
            if not expansion_plan.uses_fixed_capacity_packing:
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
        reuse_intermediate_state: IntermediateRoundState | None = None,
        mirror_kv_common_attn_metadata: CommonAttentionMetadata | None = None,
    ) -> DitRoundVerification:
        """Pivot intermediate verify.

        ``draft_like``: pass ``mirror_kv_common_attn_metadata=None``; base CAD is
        ``base_common_attn_metadata`` (same step geometry as the draft path).

        ``mirror_frontier``: optional ``mirror_kv_common_attn_metadata`` replaces the
        **base** CAD only (see ``AdaptiveSpechiveProposer.verify_chunk_with_inter_verifier``).
        """
        # When topk expansion is active, prefix_rows is expanded (length P)
        # while the base tensors / CAD still reflect the origin batch (B).
        # Expand frontier inputs so the intermediate verifier sees matching
        # batch dimensions.
        plan = self._active_pivot_expansion_plan
        if plan is not None and len(prefix_rows) != int(
            base_common_attn_metadata.batch_size()
        ):
            (
                base_target_token_ids,
                base_target_positions,
                base_target_hidden_states,
                base_next_token_ids,
                base_common_attn_metadata,
                base_num_rejected_tokens_gpu,
            ) = self._expand_tail_proposer_frontier_for_plan(
                plan=plan,
                base_target_token_ids=base_target_token_ids,
                base_target_positions=base_target_positions,
                base_target_hidden_states=base_target_hidden_states,
                base_next_token_ids=base_next_token_ids,
                base_common_attn_metadata=base_common_attn_metadata,
                base_num_rejected_tokens_gpu=base_num_rejected_tokens_gpu,
            )
        verifier_hidden_states = base_target_hidden_states
        verifier_round_state = None
        if self._pivot_mode.verification_pipeline in (
            "intermediate_then_target",
            "intermediate_tree_then_target_tree",
        ):
            if reuse_intermediate_state is not None:
                verifier_round_state = reuse_intermediate_state
                assert verifier_round_state.hidden_bundle is not None
                verifier_hidden_states = verifier_round_state.hidden_bundle.hidden_states
            else:
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
        verify_prefab = None
        if verifier_round_state is not None and verifier_round_state.hidden_bundle is not None:
            verify_prefab = verifier_round_state.hidden_bundle.prefix_prefab
        if mirror_kv_common_attn_metadata is not None:
            verify_prefab = None
        if verify_prefab is not None:
            logits_flat, bonus_logits = verify_intermediate_chunk_with_prefix_prefab(
                self._intermediate,
                prefix_prefab=verify_prefab,
                base_num_rejected_tokens_gpu=base_num_rejected_tokens_gpu,
                draft_tokens=candidate_tokens,
            )
        else:
            cad_verify = (
                mirror_kv_common_attn_metadata
                if mirror_kv_common_attn_metadata is not None
                else base_common_attn_metadata
            )
            logits_flat, bonus_logits = _verify_chunk_with_prefix(
                self._intermediate,
                cad=cad_verify,
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
        ).to(torch.int32)
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
        if _spechive_debug_enabled() and batch_size > 0:
            accepted_per_pos = [0] * chunk_len
            for accepted in accepted_lens:
                for pos in range(min(int(accepted), chunk_len)):
                    accepted_per_pos[pos] += 1
            accepted_rates = [
                accepted_per_pos[pos] / batch_size for pos in range(chunk_len)
            ]
            logger.info(
                "PIVOT_DEBUG inter_accept_rate_per_pos: rows=%d chunk_len=%d rates=%s",
                batch_size,
                chunk_len,
                [round(rate, 4) for rate in accepted_rates],
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
            self._staged_hybrid_bundle = bundle
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
        expand_ok = batch_size >= self._min_batch_for_expansion
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
                enable_topk_expansion=expand_ok,
            )
        num_out_rows = int(out.shape[0])
        source_stage_2d = torch.ones(
            num_out_rows, self._L, device=out.device, dtype=torch.int32,
        )
        if self._L > 0:
            source_stage_2d[:, 0] = 0
        row_req_ids = _pivot_hybrid_bundle_row_req_ids(
            self.runner,
            batch_size,
            num_out_rows,
            expansion_plan,
        )
        _ba_prof = (
            getattr(self.runner, "_spec_profiler", None)
            if self.runner is not None
            else None
        )
        if _ba_prof is not None:
            _ba_prof.start_stage("bundle_assemble", invocation_idx=0)
        self._staged_hybrid_bundle = _build_hybrid_bundle_from_rows(
            out,
            mode="pivot",
            draft_probs=draft_probs_flat,
            source_stage_2d=source_stage_2d,
            bundle_row_req_ids=row_req_ids,
        )
        if _ba_prof is not None:
            _ba_prof.end_stage("bundle_assemble", invocation_idx=0)
        if expansion_plan is not None:
            self._active_pivot_expansion_plan = expansion_plan
            self._is_waiting_for_target_collapse = self._pivot_spechive
            _, tree_plan = self._flatten_family_trees_for_verification(
                proposal_tokens=out,
                expansion_plan=expansion_plan,
            )
            self._pending_tree_plan = tree_plan
            self._staged_hybrid_bundle = dataclass_replace(
                self._staged_hybrid_bundle,
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
        _phb = self._staged_hybrid_bundle
        if _spechive_debug_enabled():
            logger.info(
                "PIVOT_DEBUG stage: origin_batch=%d bundle_rows=%d has_plan=%s expanded_batch=%s",
                batch_size,
                len(_phb.num_draft_tokens) if _phb is not None else 0,
                expansion_plan is not None,
                expansion_plan.expanded_batch_size if expansion_plan is not None else None,
            )
        self.clear_draft_probs()
        if expansion_plan is None:
            return _collapse_draft_rows_for_scheduler(out, expansion_plan, batch_size)
        _cl_prof = (
            getattr(self.runner, "_spec_profiler", None)
            if self.runner is not None
            else None
        )
        if _cl_prof is not None:
            _cl_prof.start_stage("expand_collapse", invocation_idx=1)
        collapsed = _collapse_draft_rows_for_scheduler(
            out, expansion_plan, batch_size
        )
        if _cl_prof is not None:
            _cl_prof.end_stage("expand_collapse", invocation_idx=1)
        return collapsed

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
