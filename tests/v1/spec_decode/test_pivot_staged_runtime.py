# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.v1.spec_decode.pivot import PivotProposer
from vllm.v1.spec_decode.spec_stage_ops import (
    build_family_flatten_order,
    collapse_family_paths_to_origin,
    collapse_family_tree_sampled_to_family_paths,
    collapse_pivot_expanded_sampled_to_origin,
    get_unselected_cleanup_rows,
    remap_hybrid_bundle_rows_for_metadata,
    select_pivot_expanded_rows_to_origin,
    validate_root_only_pivot_expansion,
)
from vllm.v1.spec_decode.spec_stage_runtime import (
    EagleTreeTemplate,
    FamilyTreeReduceResult,
    HybridProposalBundle,
    PivotExpandedTreePlan,
    PivotExpansionFamily,
    PivotExpansionPlan,
    PivotTreeFamily,
    RootTopKInfo,
    StagedHiddenStateBundle,
    expand_hybrid_bundle_for_pivot_expansion,
    pivot_expansion_indices_fit_prepare_batch,
)


def test_build_expanded_root_families_one_entry_per_packed_row() -> None:
    """PR2: tree flatten expects ``len(PivotTreeFamily) == P`` (packed rows)."""
    proposer = object.__new__(PivotProposer)
    B, P = 2, 4
    plan = PivotExpansionPlan(
        expanded_to_origin=[0, 1, 0, 0],
        families=[
            PivotExpansionFamily(
                origin_row=0,
                expanded_rows=[0, 2],
                candidate_ranks=[0, 1],
                first_token_ids=[7, 8],
                first_token_probs=[0.6, 0.4],
            )
        ],
        expanded_batch_size=P,
        origin_batch_size=B,
        packed_batch_size=P,
        packed_to_origin=[0, 1, -1, -1],
        packed_sm_origin=[0, 1, 0, 0],
        packed_row_is_active=[True, True, True, False],
        packed_row_is_base=[True, True, False, False],
        packed_row_family_rank=[0, 0, 1, -1],
        origin_to_base_row=[0, 1],
        origin_to_family_rows=[[2], []],
        uses_fixed_capacity_packing=True,
    )
    roots = proposer._build_expanded_root_families(expansion_plan=plan)
    assert len(roots) == P
    assert roots[0].root_token_id == 7
    assert roots[2].root_token_id == 8
    assert roots[3].root_token_id == 0


def test_remap_hybrid_bundle_reorders_rows_by_req_id() -> None:
    plan = PivotExpansionPlan(
        expanded_to_origin=[0, 1],
        families=[],
        expanded_batch_size=2,
        origin_batch_size=2,
        packed_batch_size=2,
        packed_to_origin=[0, 1],
        packed_sm_origin=[0, 1],
        packed_row_is_active=[True, True],
        packed_row_is_base=[True, True],
        packed_row_family_rank=[0, 0],
        origin_to_base_row=[0, 1],
        origin_to_family_rows=[[], []],
        uses_fixed_capacity_packing=True,
    )
    bundle = HybridProposalBundle(
        draft_token_ids=torch.tensor([1, 2, 3, 4], dtype=torch.int32),
        draft_probs=None,
        num_draft_tokens=[2, 2],
        cu_num_draft_tokens=torch.tensor([2, 4], dtype=torch.int32),
        max_spec_len=2,
        mode="pivot",
        expansion_plan=plan,
        bundle_row_req_ids=("req_b", "req_a"),
    )
    out, info = remap_hybrid_bundle_rows_for_metadata(bundle, ["req_a", "req_b"])
    assert info is None
    assert out is not None
    assert list(out.draft_token_ids.tolist()) == [3, 4, 1, 2]
    assert out.bundle_row_req_ids == ("req_a", "req_b")
    assert out.expansion_plan is not None
    assert list(out.expansion_plan.expanded_to_origin) == [1, 0]
    assert list(out.expansion_plan.packed_sm_origin or []) == [1, 0]


def test_remap_hybrid_bundle_subset_survivor_rows() -> None:
    """Batch shrink: keep metadata rows that are a strict subset of bundle rows."""
    bundle = HybridProposalBundle(
        draft_token_ids=torch.tensor([10, 20, 30, 40], dtype=torch.int32),
        draft_probs=None,
        num_draft_tokens=[1, 1, 1, 1],
        cu_num_draft_tokens=torch.tensor([1, 2, 3, 4], dtype=torch.int32),
        max_spec_len=1,
        mode="pivot",
        expansion_plan=None,
        bundle_row_req_ids=("r_a", "r_b", "r_c", "r_d"),
    )
    out, info = remap_hybrid_bundle_rows_for_metadata(bundle, ["r_b", "r_d"])
    assert info == "subset_survivor_remap"
    assert out is not None
    assert list(out.draft_token_ids.tolist()) == [20, 40]
    assert out.bundle_row_req_ids == ("r_b", "r_d")


def test_expand_hybrid_bundle_keeps_probs_when_fixed_capacity_and_families() -> None:
    """PR3: do not strip ``draft_probs`` solely because ``families`` is non-empty."""
    B, P = 2, 4
    plan = PivotExpansionPlan(
        expanded_to_origin=[0, 1, 0, 0],
        families=[
            PivotExpansionFamily(
                origin_row=0,
                expanded_rows=[0],
                candidate_ranks=[0],
                first_token_ids=[1],
                first_token_probs=[1.0],
            )
        ],
        expanded_batch_size=P,
        origin_batch_size=B,
        packed_batch_size=P,
        packed_to_origin=[0, 1, -1, -1],
        packed_sm_origin=[0, 1, 0, 0],
        packed_row_is_active=[True, True, False, False],
        packed_row_is_base=[True, True, False, False],
        packed_row_family_rank=[0, 0, -1, -1],
        origin_to_base_row=[0, 1],
        origin_to_family_rows=[[], []],
        uses_fixed_capacity_packing=True,
    )
    lengths = [1, 1, 0, 0]
    draft_tok = torch.zeros((2,), dtype=torch.int32)
    cu = torch.tensor([1, 2], dtype=torch.int32)
    probs = torch.randn(2, 16)
    bundle = HybridProposalBundle(
        draft_token_ids=draft_tok,
        draft_probs=probs,
        num_draft_tokens=lengths,
        cu_num_draft_tokens=cu,
        max_spec_len=1,
        mode="pivot",
        expansion_plan=plan,
    )
    out = expand_hybrid_bundle_for_pivot_expansion(bundle)
    assert out.draft_probs is not None


def test_pivot_expansion_indices_fit_prepare_batch() -> None:
    plan = PivotExpansionPlan(
        expanded_to_origin=[0, 1],
        families=[],
        expanded_batch_size=2,
    )
    assert pivot_expansion_indices_fit_prepare_batch(plan, num_reqs=2)
    assert not pivot_expansion_indices_fit_prepare_batch(plan, num_reqs=1)
    assert not pivot_expansion_indices_fit_prepare_batch(plan, num_reqs=0)


def test_pivot_expansion_indices_fit_fixed_capacity_packed() -> None:
    """``packed_to_origin`` may use -1; ``packed_sm_origin`` must stay in-range."""
    B, P = 2, 4
    plan = PivotExpansionPlan(
        expanded_to_origin=[0, 1, 0, 0],
        families=[],
        expanded_batch_size=P,
        origin_batch_size=B,
        packed_batch_size=P,
        packed_to_origin=[0, 1, -1, -1],
        packed_sm_origin=[0, 1, 0, 0],
        packed_row_is_active=[True, True, False, False],
        packed_row_is_base=[True, True, False, False],
        packed_row_family_rank=[0, 0, -1, -1],
        origin_to_base_row=[0, 1],
        origin_to_family_rows=[[], []],
        uses_fixed_capacity_packing=True,
    )
    assert pivot_expansion_indices_fit_prepare_batch(
        plan, num_reqs=B, expected_packed_size=P
    )
    assert not pivot_expansion_indices_fit_prepare_batch(
        plan, num_reqs=1, expected_packed_size=P
    )
    assert not pivot_expansion_indices_fit_prepare_batch(
        plan, num_reqs=B, expected_packed_size=P + 1
    )


def test_pivot_expansion_indices_fit_rejects_inconsistent_inactive_sentinel() -> None:
    B, P = 2, 4
    bad = PivotExpansionPlan(
        expanded_to_origin=[0, 1, 0, 0],
        families=[],
        expanded_batch_size=P,
        origin_batch_size=B,
        packed_batch_size=P,
        packed_to_origin=[0, 1, 0, -1],
        packed_sm_origin=[0, 1, 0, 0],
        packed_row_is_active=[True, True, False, False],
        packed_row_is_base=[True, True, False, False],
        packed_row_family_rank=[0, 0, -1, -1],
        origin_to_base_row=[0, 1],
        origin_to_family_rows=[[], []],
        uses_fixed_capacity_packing=True,
    )
    assert not pivot_expansion_indices_fit_prepare_batch(
        bad, num_reqs=B, expected_packed_size=P
    )


def test_pivot_expansion_indices_fit_rejects_bad_origin_to_base_row() -> None:
    B, P = 2, 4
    bad = PivotExpansionPlan(
        expanded_to_origin=[0, 1, 0, 0],
        families=[],
        expanded_batch_size=P,
        origin_batch_size=B,
        packed_batch_size=P,
        packed_to_origin=[0, 1, -1, -1],
        packed_sm_origin=[0, 1, 0, 0],
        packed_row_is_active=[True, True, False, False],
        packed_row_is_base=[True, True, False, False],
        packed_row_family_rank=[0, 0, -1, -1],
        origin_to_base_row=[0, 0],
        origin_to_family_rows=[[], []],
        uses_fixed_capacity_packing=True,
    )
    assert not pivot_expansion_indices_fit_prepare_batch(
        bad, num_reqs=B, expected_packed_size=P
    )


def test_get_unselected_cleanup_rows_skips_inactive_family_rows() -> None:
    """Inactive packed indices must not appear in cleanup even if listed on a family."""
    plan = PivotExpansionPlan(
        expanded_to_origin=[0, 0],
        families=[
            PivotExpansionFamily(
                origin_row=0,
                expanded_rows=[0, 1],
                candidate_ranks=[0, 1],
                first_token_ids=[10, 20],
                first_token_probs=[0.5, 0.5],
            )
        ],
        expanded_batch_size=2,
        origin_batch_size=1,
        packed_batch_size=2,
        packed_to_origin=[0, -1],
        packed_sm_origin=[0, 0],
        packed_row_is_active=[True, False],
        packed_row_is_base=[True, False],
        packed_row_family_rank=[0, -1],
        origin_to_base_row=[0],
        origin_to_family_rows=[[]],
        uses_fixed_capacity_packing=True,
    )
    assert pivot_expansion_indices_fit_prepare_batch(plan, num_reqs=1, expected_packed_size=2)
    assert get_unselected_cleanup_rows(expansion_plan=plan, selected_rows=[0]) == []


def test_collapse_pivot_without_family_uses_origin_to_base_row() -> None:
    """Inactive packed rows must not affect origin count (use ``origin_batch_size``)."""
    B, P = 2, 4
    plan = PivotExpansionPlan(
        expanded_to_origin=[0, 1, 0, 0],
        families=[],
        expanded_batch_size=P,
        origin_batch_size=B,
        packed_batch_size=P,
        packed_to_origin=[0, 1, -1, -1],
        packed_sm_origin=[0, 1, 0, 0],
        packed_row_is_active=[True, True, False, False],
        packed_row_is_base=[True, True, False, False],
        packed_row_family_rank=[0, 0, -1, -1],
        origin_to_base_row=[0, 1],
        origin_to_family_rows=[[], []],
        uses_fixed_capacity_packing=True,
    )
    sampled = torch.tensor([[10], [20], [99], [99]], dtype=torch.int32)
    out = collapse_pivot_expanded_sampled_to_origin(sampled, plan)
    assert out.shape == (B, 1)
    assert int(out[0, 0].item()) == 10
    assert int(out[1, 0].item()) == 20
    selected = select_pivot_expanded_rows_to_origin(sampled, plan)
    out_passthrough = collapse_pivot_expanded_sampled_to_origin(
        sampled, plan, selected_rows=selected
    )
    assert torch.equal(out, out_passthrough)


def test_validate_root_only_pivot_expansion_rejects_non_root_prefix() -> None:
    plan = PivotExpansionPlan(
        expanded_to_origin=[0, 0],
        families=[
            PivotExpansionFamily(
                origin_row=0,
                expanded_rows=[0, 1],
                candidate_ranks=[0, 1],
                first_token_ids=[1, 2],
                first_token_probs=[0.6, 0.4],
            )
        ],
        expanded_batch_size=2,
    )
    result = validate_root_only_pivot_expansion(
        prefix_rows=[[11]],
        expansion_plan=plan,
    )
    assert not result.ok


def test_validate_root_only_pivot_expansion_dense_prefix() -> None:
    plan = PivotExpansionPlan(
        expanded_to_origin=[0, 0],
        families=[
            PivotExpansionFamily(
                origin_row=0,
                expanded_rows=[0, 1],
                candidate_ranks=[0, 1],
                first_token_ids=[1, 2],
                first_token_probs=[0.6, 0.4],
            )
        ],
        expanded_batch_size=2,
    )
    lens = torch.tensor([0, 0], dtype=torch.int32)
    ok = validate_root_only_pivot_expansion(
        prefix_lengths_tensor=lens,
        prefix_num_rows=2,
        expansion_plan=plan,
    )
    assert ok.ok
    lens2 = torch.tensor([0, 1], dtype=torch.int32)
    bad = validate_root_only_pivot_expansion(
        prefix_lengths_tensor=lens2,
        prefix_num_rows=2,
        expansion_plan=plan,
    )
    assert not bad.ok


def test_get_unselected_cleanup_rows_returns_unselected_family_rows() -> None:
    plan = PivotExpansionPlan(
        expanded_to_origin=[0, 0, 1],
        families=[
            PivotExpansionFamily(
                origin_row=0,
                expanded_rows=[0, 1],
                candidate_ranks=[0, 1],
                first_token_ids=[10, 20],
                first_token_probs=[0.7, 0.3],
            )
        ],
        expanded_batch_size=3,
    )
    assert get_unselected_cleanup_rows(expansion_plan=plan, selected_rows=[1]) == [0]


def test_build_pivot_expansion_plan_skips_when_probs_and_root_topk_missing() -> None:
    """If pivot step probs and root top-k are unavailable, skip expansion."""
    proposer = object.__new__(PivotProposer)
    proposer._topk_selection = 5
    initial = torch.tensor([[42], [43]], dtype=torch.int32)
    ep, eprob, plan = proposer._build_pivot_expansion_plan(
        initial_pivots=initial,
        pivot_probs=None,
        enable_topk_expansion=True,
        root_topk_info=None,
    )
    assert torch.equal(ep, initial)
    assert eprob is None
    assert plan is None


def test_build_pivot_expansion_plan_from_root_topk_when_probs_missing() -> None:
    """RootTopKInfo alone is enough for fixed-capacity expansion (parallel drafting path)."""
    proposer = object.__new__(PivotProposer)
    proposer._topk_selection = 4
    proposer._expansion_pct = 0.5
    proposer._pivot_use_eagle_tree = False
    # B=2, num_expand=ceil(2*0.5)=1, K=4, p_extra=3, P=5
    initial = torch.tensor([[10], [20]], dtype=torch.int32)
    root_topk = RootTopKInfo(
        topk_token_ids=torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]]),
        topk_probs=torch.tensor(
            [[0.7, 0.15, 0.1, 0.05], [0.6, 0.2, 0.1, 0.1]], dtype=torch.float32
        ),
    )
    ep, eprob, plan = proposer._build_pivot_expansion_plan(
        initial_pivots=initial,
        pivot_probs=None,
        enable_topk_expansion=True,
        root_topk_info=root_topk,
    )
    assert plan is not None
    assert plan.expanded_batch_size == 5
    assert plan.uses_fixed_capacity_packing
    assert eprob is None
    assert ep.shape[0] == 5
    assert plan.row_gather_idx_cpu is not None
    assert plan.packed_sm_origin_t is not None
    assert torch.equal(
        plan.row_gather_idx_cpu,
        torch.tensor(plan.packed_sm_origin or [], dtype=torch.long),
    )


def test_pivot_probs_sparse_from_root_topk_scatters_per_packed_row() -> None:
    """RootTopK-only expansion still yields [P,1,V] for hybrid bundle draft_probs."""
    proposer = object.__new__(PivotProposer)
    proposer.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(get_vocab_size=lambda: 32)
    )
    plan = PivotExpansionPlan(
        expanded_to_origin=[0, 1],
        families=[],
        expanded_batch_size=2,
        origin_batch_size=2,
        packed_batch_size=2,
        packed_to_origin=[0, 1],
        packed_sm_origin=[0, 1],
        packed_row_is_active=[True, True],
        packed_row_is_base=[True, True],
        packed_row_family_rank=[0, 0],
        origin_to_base_row=[0, 1],
        origin_to_family_rows=[[], []],
        uses_fixed_capacity_packing=True,
    )
    root_topk = RootTopKInfo(
        topk_token_ids=torch.tensor([[3, 4], [5, 6]]),
        topk_probs=torch.tensor([[0.8, 0.2], [0.9, 0.1]], dtype=torch.float32),
    )
    dense = proposer._pivot_probs_sparse_from_root_topk_for_plan(
        plan, root_topk, vocab_size=32
    )
    assert dense.shape == (2, 1, 32)
    assert abs(float(dense[0, 0, 3].item()) - 0.8) < 1e-5
    assert abs(float(dense[1, 0, 5].item()) - 0.9) < 1e-5


def test_pivot_bundle_row_count_matches_metadata_rows() -> None:
    """Regression: staged hybrid bundle row count matches spec decode metadata."""
    plan = PivotExpansionPlan(
        expanded_to_origin=[0, 1, 0, 0],
        families=[],
        expanded_batch_size=4,
        origin_batch_size=2,
        packed_batch_size=4,
        packed_to_origin=[0, 1, -1, -1],
        packed_sm_origin=[0, 1, 0, 0],
        packed_row_is_active=[True, True, True, False],
        packed_row_is_base=[True, True, False, False],
        packed_row_family_rank=[0, 0, 1, -1],
        origin_to_base_row=[0, 1],
        origin_to_family_rows=[[], []],
        uses_fixed_capacity_packing=True,
    )
    bundle = HybridProposalBundle(
        draft_token_ids=torch.zeros(8, dtype=torch.int32),
        draft_probs=None,
        num_draft_tokens=[2, 2, 2, 2],
        cu_num_draft_tokens=torch.tensor([2, 4, 6, 8], dtype=torch.int32),
        max_spec_len=2,
        mode="pivot",
        expansion_plan=plan,
        bundle_row_req_ids=("a", "b", "a", "a"),
    )
    assert len(bundle.num_draft_tokens) == plan.expanded_batch_size


def test_pivot_bootstrap_hidden_source_uses_provider_in_intermediate_mode() -> None:
    proposer = object.__new__(PivotProposer)
    called = {"value": False}

    class _FakeProvider:
        def bootstrap_from_current_prefix(self, **kwargs):
            called["value"] = True
            del kwargs
            return SimpleNamespace(
                bootstrap_complete=True,
                hidden_bundle=StagedHiddenStateBundle(
                    hidden_states=torch.ones((2, 4), dtype=torch.float32),
                    aux_hidden_states=None,
                    batch_size=2,
                ),
            )

    proposer._pivot_mode = SimpleNamespace(
        proposal_engine="eagle3_head",
        hidden_state_source="intermediate",
    )
    proposer._staged_delegates = SimpleNamespace(hidden_state_provider=_FakeProvider())
    out, prefab = proposer._resolve_hidden_states_for_proposal(
        base_target_token_ids=torch.zeros((2,), dtype=torch.int32),
        base_target_positions=torch.zeros((2,), dtype=torch.int64),
        base_target_hidden_states=torch.zeros((2, 4), dtype=torch.float32),
        base_next_token_ids=torch.zeros((2,), dtype=torch.int32),
        base_common_attn_metadata=None,
        base_num_rejected_tokens_gpu=None,
        prefix_rows=[[], []],
    )
    assert called["value"]
    assert prefab is None
    assert torch.allclose(out, torch.ones((2, 4), dtype=torch.float32))


def _make_tree_plan() -> PivotExpandedTreePlan:
    template = EagleTreeTemplate(
        parent_ids=[-1, 0, 0],
        node_depths=[1, 2, 2],
        node_order=[0, 1, 2],
        leaf_ids=[1, 2],
        num_nodes=3,
    )
    families = [
        PivotTreeFamily(
            origin_row=0,
            family_id=0,
            root_rank=0,
            root_token_id=10,
            node_row_start=0,
            node_row_end=3,
        ),
        PivotTreeFamily(
            origin_row=0,
            family_id=1,
            root_rank=1,
            root_token_id=20,
            node_row_start=3,
            node_row_end=6,
        ),
    ]
    return build_family_flatten_order(
        families=families, template=template, origin_batch_size=1
    )


def test_build_family_flatten_order_is_family_major_stable() -> None:
    plan = _make_tree_plan()
    assert plan.flat_to_family_ids == [0, 0, 0, 1, 1, 1]
    assert plan.flat_to_node_ids == [0, 1, 2, 0, 1, 2]


def test_collapse_family_tree_sampled_to_family_paths_reconstructs() -> None:
    plan = _make_tree_plan()
    sampled = torch.tensor(
        [
            [101, 102, -1, 900],
            [201, 202, 203, 901],
        ],
        dtype=torch.int32,
    )
    reduced = collapse_family_tree_sampled_to_family_paths(
        sampled, plan=plan, num_draft_tokens=[3, 3]
    )
    assert reduced.accepted_lens == [2, 2]
    assert reduced.accepted_rows == [[101, 102], [201, 202]]
    assert reduced.chosen_leaf_ids[0] in (1, 2)
    assert reduced.chosen_leaf_ids[1] in (1, 2)


def test_collapse_family_tree_reconstructs_topology_path_not_row_prefix() -> None:
    template = EagleTreeTemplate(
        parent_ids=[-1, 0, 0],
        node_depths=[1, 2, 2],
        node_order=[0, 1, 2],
        leaf_ids=[1, 2],
        num_nodes=3,
    )
    families = [
        PivotTreeFamily(
            origin_row=0,
            family_id=0,
            root_rank=0,
            root_token_id=10,
            node_row_start=0,
            node_row_end=3,
        )
    ]
    plan = build_family_flatten_order(
        families=families,
        template=template,
        origin_batch_size=1,
    )
    # accepted_len_raw=3 means nodes 0,1,2 all accepted in flat order.
    # Topology-aware reconstruction picks one valid leaf path, not flat prefix.
    sampled = torch.tensor([[100, 101, 102, 999]], dtype=torch.int32)
    reduced = collapse_family_tree_sampled_to_family_paths(
        sampled, plan=plan, num_draft_tokens=[3]
    )
    assert reduced.accepted_lens[0] == 2
    assert reduced.accepted_rows[0] in ([100, 101], [100, 102])


def test_collapse_family_paths_to_origin_prefers_max_accept_len_then_rank() -> None:
    plan = _make_tree_plan()
    sampled = torch.tensor(
        [
            [111, 112, -1, 501],
            [211, 212, 213, 502],
        ],
        dtype=torch.int32,
    )
    reduced = FamilyTreeReduceResult(
        accepted_rows=[[111, 112], [211, 212, 213]],
        accepted_lens=[2, 3],
        chosen_leaf_ids=[1, 2],
        recovery_token_ids=[501, 502],
    )
    collapsed, selected = collapse_family_paths_to_origin(
        sampled, plan=plan, reduced=reduced
    )
    assert selected == [1]
    assert collapsed.shape[0] == 1
    assert int(collapsed[0, 0].item()) == 211
