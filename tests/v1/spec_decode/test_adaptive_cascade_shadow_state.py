# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for adaptive-spechive hierarchical verification semantics."""

import torch

from vllm.v1.spec_decode.spec_stage_ops import (
    validate_hybrid_bundle_draft_layout,
    validate_hierarchical_verification_source_stage_row,
    validate_hierarchical_verification_tail_len,
    validate_hierarchical_verification_tail_len_rowwise,
)
from vllm.v1.spec_decode.spec_stage_runtime import HybridProposalBundle


def test_validate_dit_tail_len_success() -> None:
    result = validate_hierarchical_verification_tail_len(
        tail_len=2, interval_tokens=2, remaining_cap=5
    )
    assert result.code == "check4_pre_target_tail_draft"
    assert result.ok is True


def test_validate_dit_tail_len_failure_code() -> None:
    result = validate_hierarchical_verification_tail_len(
        tail_len=3, interval_tokens=2, remaining_cap=1
    )
    assert result.code == "check4_pre_target_tail_draft"
    assert result.ok is False
    assert "expected=1" in result.detail


def test_validate_dit_tail_len_rowwise_matches_scalar_max() -> None:
    result = validate_hierarchical_verification_tail_len_rowwise(
        tail_len=2,
        interval_tokens=2,
        remaining_cap_per_row=[1, 2, 0],
    )
    assert result.code == "check4_pre_target_tail_draft_rowwise"
    assert result.ok is True


def test_validate_dit_tail_len_rowwise_failure() -> None:
    result = validate_hierarchical_verification_tail_len_rowwise(
        tail_len=5,
        interval_tokens=2,
        remaining_cap_per_row=[10, 10],
    )
    assert result.code == "check4_pre_target_tail_draft_rowwise"
    assert result.ok is False


def test_validate_hybrid_bundle_draft_layout_ok() -> None:
    bundle = HybridProposalBundle(
        draft_token_ids=torch.arange(6, dtype=torch.int32),
        draft_probs=None,
        num_draft_tokens=[2, 2, 2],
        cu_num_draft_tokens=torch.tensor([2, 4, 6], dtype=torch.int32),
        max_spec_len=3,
        mode="hierarchical_verification",
    )
    r = validate_hybrid_bundle_draft_layout(bundle, expected_num_rows=3)
    assert r.code == "check_hv_hybrid_bundle_draft_layout"
    assert r.ok is True


def test_validate_hybrid_bundle_draft_layout_bad_cu() -> None:
    bundle = HybridProposalBundle(
        draft_token_ids=torch.arange(6, dtype=torch.int32),
        draft_probs=None,
        num_draft_tokens=[2, 2, 2],
        cu_num_draft_tokens=torch.tensor([2, 4, 5], dtype=torch.int32),
        max_spec_len=3,
        mode="hierarchical_verification",
    )
    r = validate_hybrid_bundle_draft_layout(bundle, expected_num_rows=3)
    assert r.ok is False


def test_validate_dit_source_stage_row_success() -> None:
    result = validate_hierarchical_verification_source_stage_row(
        source_stage_row=[0, 0, 1], bundle_len=3
    )
    assert result.code == "check8_source_stage_consistency"
    assert result.ok is True


def test_validate_dit_source_stage_row_failure_code() -> None:
    result = validate_hierarchical_verification_source_stage_row(
        source_stage_row=[0, 2, 1], bundle_len=3
    )
    assert result.code == "check8_source_stage_consistency"
    assert result.ok is False
    assert "has_only_known_stages=False" in result.detail
