# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.v1.spec_decode.pivot import PivotProposer
from vllm.v1.spec_decode.spec_stage_ops import (
    build_family_flatten_order,
    collapse_family_paths_to_origin,
    collapse_family_tree_sampled_to_family_paths,
    get_unselected_cleanup_rows,
    validate_root_only_pivot_expansion,
)
from vllm.v1.spec_decode.spec_stage_runtime import (
    EagleTreeTemplate,
    FamilyTreeReduceResult,
    PivotExpandedTreePlan,
    PivotExpansionFamily,
    PivotExpansionPlan,
    PivotTreeFamily,
    StagedHiddenStateBundle,
)


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


def test_build_pivot_expansion_plan_skips_when_probs_missing_but_topk_enabled() -> None:
    """If pivot step probs are unavailable, skip expansion without crashing."""
    proposer = object.__new__(PivotProposer)
    proposer._topk_selection = 5
    initial = torch.tensor([[42], [43]], dtype=torch.int32)
    ep, eprob, plan = proposer._build_pivot_expansion_plan(
        initial_pivots=initial,
        pivot_probs=None,
        enable_topk_expansion=True,
    )
    assert torch.equal(ep, initial)
    assert eprob is None
    assert plan is None


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
    out = proposer._resolve_hidden_states_for_proposal(
        base_target_token_ids=torch.zeros((2,), dtype=torch.int32),
        base_target_positions=torch.zeros((2,), dtype=torch.int64),
        base_target_hidden_states=torch.zeros((2, 4), dtype=torch.float32),
        base_next_token_ids=torch.zeros((2,), dtype=torch.int32),
        base_common_attn_metadata=None,
        base_num_rejected_tokens_gpu=None,
        prefix_rows=[[], []],
    )
    assert called["value"]
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
