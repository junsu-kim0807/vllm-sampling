# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TETRIS capacity semantics: full grid => no trimming vs vanilla."""

from __future__ import annotations

import pytest
import torch

try:
    from vllm.v1.spec_decode.tetris import select_proposals
except ImportError:
    select_proposals = None  # type: ignore[misc, assignment]


def _has_torch_scatter() -> bool:
    try:
        import torch_scatter  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    select_proposals is None or not _has_torch_scatter(),
    reason="tetris.select_proposals requires torch-scatter",
)
def test_full_capacity_selects_max_depth_every_request() -> None:
    """When capacity == B*K, every (request, depth) cell can be chosen; scatter_max
    per request is K — same draft lengths as vanilla spec-decode."""
    batch_size, k = 5, 4
    # Random finite logprobs (cumulative sums stay ordered for the test purpose).
    draft_token_logprobs = torch.randn(batch_size, k)
    num_draft_tokens = [k] * batch_size
    lengths = select_proposals(
        capacity=batch_size * k,
        draft_token_logprobs=draft_token_logprobs,
        num_draft_tokens=num_draft_tokens,
    )
    assert lengths == [k] * batch_size


@pytest.mark.skipif(
    select_proposals is None or not _has_torch_scatter(),
    reason="tetris.select_proposals requires torch-scatter",
)
def test_single_capacity_slot_affects_at_most_one_request() -> None:
    """With capacity 1, only one (request, depth) pair wins; others should not
    all retain full length K when K > 1."""
    batch_size, k = 6, 4
    draft_token_logprobs = torch.randn(batch_size, k)
    num_draft_tokens = [k] * batch_size
    lengths = select_proposals(
        capacity=1,
        draft_token_logprobs=draft_token_logprobs,
        num_draft_tokens=num_draft_tokens,
    )
    assert sum(1 for L in lengths if L > 0) == 1
    assert sum(lengths) <= k
