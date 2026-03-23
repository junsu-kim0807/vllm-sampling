# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for adaptive-spechive hierarchical verification semantics."""

from vllm.v1.spec_decode.spec_stage_ops import (
    validate_hierarchical_verification_source_stage_row,
    validate_hierarchical_verification_tail_len,
)


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
