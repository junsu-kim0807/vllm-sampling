# SPDX-License-Identifier: Apache-2.0
"""Unit tests for staged hidden bundle expansion (pivot row packing)."""

import torch

from vllm.v1.spec_decode.spec_stage_ops import expand_staged_hidden_bundle_for_pivot_plan
from vllm.v1.spec_decode.spec_stage_runtime import PivotExpansionPlan, StagedHiddenStateBundle


def test_expand_staged_hidden_bundle_no_prefab_is_identity() -> None:
    b = StagedHiddenStateBundle(
        hidden_states=torch.zeros(1, 4),
        aux_hidden_states=None,
        batch_size=1,
        owns_provisional_frontier=True,
        prefix_prefab=None,
    )
    plan = PivotExpansionPlan(
        expanded_to_origin=[0, 0],
        families=[],
        expanded_batch_size=2,
        origin_batch_size=1,
    )
    out = expand_staged_hidden_bundle_for_pivot_plan(b, plan)
    assert out is b
    assert out.prefix_prefab is None
