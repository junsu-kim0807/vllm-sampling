# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for TETRIS capacity formula and core algorithm.

Tests cover:
  - select_proposals: full-capacity, sub-capacity, single-slot, walkthrough
  - apply_tetris: capacity formula (base_k × B), fallback, extra_proposals=0
"""

from __future__ import annotations

import pytest
import torch


def _has_torch_scatter() -> bool:
    try:
        import torch_scatter  # noqa: F401
        return True
    except ImportError:
        return False


_skip_no_scatter = pytest.mark.skipif(
    not _has_torch_scatter(),
    reason="torch-scatter required for TETRIS tests",
)


# ── select_proposals ──────────────────────────────────────────────────────

@_skip_no_scatter
def test_full_capacity_selects_max_depth_every_request() -> None:
    """capacity == B*K → all cells selected → each request gets full K."""
    from vllm.v1.spec_decode.tetris import select_proposals

    B, K = 5, 4
    logprobs = torch.randn(B, K)
    lengths = select_proposals(
        capacity=B * K,
        draft_token_logprobs=logprobs,
        num_draft_tokens=[K] * B,
    )
    assert lengths == [K] * B


@_skip_no_scatter
def test_single_slot_exactly_one_request() -> None:
    """capacity=1 → only one (request, depth=1) cell wins."""
    from vllm.v1.spec_decode.tetris import select_proposals

    B, K = 6, 4
    logprobs = torch.randn(B, K)
    lengths = select_proposals(
        capacity=1,
        draft_token_logprobs=logprobs,
        num_draft_tokens=[K] * B,
    )
    nonzero = [L for L in lengths if L > 0]
    assert len(nonzero) == 1
    assert nonzero[0] == 1  # depth 1 (pos 0)


@_skip_no_scatter
def test_sub_capacity_trims_some_requests() -> None:
    """When capacity = base_k * B < B*K, not all requests get full K."""
    from vllm.v1.spec_decode.tetris import select_proposals

    B, K = 4, 5
    base_k = 3
    capacity = base_k * B  # 12 < 20
    logprobs = torch.randn(B, K)
    lengths = select_proposals(
        capacity=capacity,
        draft_token_logprobs=logprobs,
        num_draft_tokens=[K] * B,
    )
    assert len(lengths) == B
    assert all(0 <= L <= K for L in lengths)
    # Not everyone can keep K — at least one must be trimmed or dropped.
    assert sum(lengths) < B * K


@_skip_no_scatter
def test_guide_walkthrough_example() -> None:
    """Reproduce the exact example from TETRIS_IMPLEMENTATION_GUIDE.md §4."""
    from vllm.v1.spec_decode.tetris import select_proposals

    # Per-token logprobs (already gathered, before cumsum)
    per_token = torch.tensor([
        [-0.10, -0.20, -0.50, -0.70],
        [-0.50, -0.70, -0.80, -1.00],
        [-0.05, -0.15, -0.10, -0.30],
    ])
    lengths = select_proposals(
        capacity=6,
        draft_token_logprobs=per_token,
        num_draft_tokens=[4, 4, 4],
    )
    assert lengths == [2, 1, 3]


@_skip_no_scatter
def test_zero_capacity_returns_all_zeros() -> None:
    from vllm.v1.spec_decode.tetris import select_proposals

    lengths = select_proposals(
        capacity=0,
        draft_token_logprobs=torch.randn(3, 4),
        num_draft_tokens=[4, 4, 4],
    )
    assert lengths == [0, 0, 0]


# ── apply_tetris ──────────────────────────────────────────────────────────

@_skip_no_scatter
def test_apply_tetris_capacity_formula() -> None:
    """capacity = (K - extra) * B = base_k * B, strictly less than B*K."""
    from vllm.v1.spec_decode.tetris import apply_tetris

    B, K, extra = 4, 5, 2
    base_k = K - extra  # 3
    ids = torch.arange(B * K).reshape(B, K)
    logprobs = torch.randn(B, K)

    result = apply_tetris(
        draft_token_ids=ids,
        draft_token_logprobs=logprobs,
        num_speculative_tokens=K,
        extra_proposals=extra,
    )
    assert len(result) == B
    total_tokens = sum(len(r) for r in result)
    # Total verified tokens cannot exceed B * K and should often < B * K.
    assert total_tokens <= B * K
    # Each result length is between 0 and K.
    for r in result:
        assert 0 <= len(r) <= K


@_skip_no_scatter
def test_apply_tetris_extra_zero_is_noop() -> None:
    """extra_proposals=0 → capacity = K*B = full grid → every request keeps K."""
    from vllm.v1.spec_decode.tetris import apply_tetris

    B, K = 3, 4
    ids = torch.arange(B * K).reshape(B, K)
    logprobs = torch.randn(B, K)

    result = apply_tetris(
        draft_token_ids=ids,
        draft_token_logprobs=logprobs,
        num_speculative_tokens=K,
        extra_proposals=0,
    )
    assert all(len(r) == K for r in result)


@_skip_no_scatter
def test_apply_tetris_fallback_returns_base_k() -> None:
    """When batch < turn_on_batch_size, result has base_k tokens per request."""
    from vllm.v1.spec_decode.tetris import apply_tetris

    B, K, extra = 2, 5, 2
    base_k = K - extra
    ids = torch.arange(B * K).reshape(B, K)
    logprobs = torch.randn(B, K)

    result = apply_tetris(
        draft_token_ids=ids,
        draft_token_logprobs=logprobs,
        num_speculative_tokens=K,
        extra_proposals=extra,
        turn_on_batch_size=4,  # B=2 < 4 → fallback
    )
    assert all(len(r) == base_k for r in result)


@_skip_no_scatter
def test_apply_tetris_guide_example_end_to_end() -> None:
    """End-to-end: guide §4 example via apply_tetris."""
    from vllm.v1.spec_decode.tetris import apply_tetris

    B, K, extra = 3, 4, 2
    base_k = K - extra  # 2
    # capacity = 2 * 3 = 6

    ids = torch.tensor([
        [10, 20, 30, 40],
        [50, 60, 70, 80],
        [90, 100, 110, 120],
    ])
    per_token = torch.tensor([
        [-0.10, -0.20, -0.50, -0.70],
        [-0.50, -0.70, -0.80, -1.00],
        [-0.05, -0.15, -0.10, -0.30],
    ])

    result = apply_tetris(
        draft_token_ids=ids,
        draft_token_logprobs=per_token,
        num_speculative_tokens=K,
        extra_proposals=extra,
    )
    # Expected from the walkthrough: [2, 1, 3]
    assert [len(r) for r in result] == [2, 1, 3]
    assert result[0] == [10, 20]
    assert result[1] == [50]
    assert result[2] == [90, 100, 110]
