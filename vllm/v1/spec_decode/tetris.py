# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TETRIS: Optimal Draft Token Selection for Batch Speculative Decoding.

Paper: https://arxiv.org/pdf/2502.15197
Published at ACL 2025 (main track).

Authors: Zhaoxuan Wu, Zijian Zhou, Arun Verma, Alok Prakash,
         Daniela Rus, Bryan Kian Hsiang Low.

This module implements TETRIS for vLLM's V1 engine.
The core algorithm selects, for every scheduler step, which subset of
draft tokens across the entire batch to actually send to the target model
for verification, subject to a total *capacity* constraint.

Key formula (from the paper):
    actual_draft_len (K) = base_k + extra_proposals
    capacity = base_k × batch_size = (K − extra_proposals) × batch_size

The drafter produces K tokens per request, but the target model only
verifies capacity = base_k × B of them.  TETRIS decides *which* prefix
depths to keep per request so the expected accepted tokens is maximised.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_TETRIS_DEBUG_ENV = "VLLM_TETRIS_DEBUG"


def _tetris_debug_enabled() -> bool:
    return os.getenv(_TETRIS_DEBUG_ENV, "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

# Lazy import so the rest of vLLM loads even when torch-scatter is absent.
_scatter_max = None


def _get_scatter_max():
    global _scatter_max
    if _scatter_max is None:
        try:
            from torch_scatter import scatter_max as _sm
            _scatter_max = _sm
        except ImportError as exc:
            raise ImportError(
                "TETRIS requires the `torch-scatter` library. "
                "Install it with:\n"
                "  pip install torch-scatter "
                "-f https://data.pyg.org/whl/torch-2.9.0+<CUDA>.html\n"
                "where <CUDA> matches your CUDA version "
                "(e.g. cu124 for CUDA 12.4)."
            ) from exc
    return _scatter_max


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def select_proposals(
    capacity: int,
    draft_token_logprobs: torch.Tensor,
    num_draft_tokens: list[int],
) -> list[int]:
    """TETRIS optimal draft token selection.

    Given per-request, per-position log-probabilities of the selected draft
    tokens, find the assignment of *at most* ``capacity`` verification slots
    to (request, depth) positions that maximises the expected number of
    accepted tokens.

    Args:
        capacity: Total number of draft-token verification slots for the
            batch.  Must satisfy ``0 < capacity <= B * K``.
        draft_token_logprobs: Float tensor of shape
            ``[batch_size, max_spec_len]`` containing the log-probability
            (under the draft model) of each selected draft token.
        num_draft_tokens: List of length ``batch_size`` with the number of
            valid draft tokens per request.

    Returns:
        List of length ``batch_size`` with TETRIS-optimal proposal lengths.
    """
    scatter_max = _get_scatter_max()

    batch_size, max_spec_len = draft_token_logprobs.shape
    device = draft_token_logprobs.device

    if batch_size == 0 or capacity <= 0:
        return [0] * batch_size

    # 1. Mask padding positions
    valid_mask = torch.zeros(
        batch_size, max_spec_len, dtype=torch.bool, device=device
    )
    for i, n in enumerate(num_draft_tokens):
        if n > 0:
            valid_mask[i, :n] = True

    masked_logprobs = draft_token_logprobs.masked_fill(
        ~valid_mask, float("-inf")
    )

    # 2. Cumulative log-probability (log of joint acceptance prob)
    cumulative_logprobs = masked_logprobs.cumsum(dim=-1)  # [B, K]

    # 3. Flatten and top-k
    flat = cumulative_logprobs.flatten()  # [B * K]
    effective_capacity = min(capacity, int(valid_mask.sum().item()))
    if effective_capacity == 0:
        return [0] * batch_size

    _, best_indices = torch.topk(flat, effective_capacity, largest=True)

    request_indices = best_indices // max_spec_len       # which request
    token_lengths = (best_indices % max_spec_len) + 1    # depth + 1 = length

    # 4. Per-request maximum selected depth (scatter_max)
    best_frontier, _ = scatter_max(
        token_lengths,
        request_indices,
        dim_size=batch_size,
    )

    return best_frontier.tolist()


# ---------------------------------------------------------------------------
# High-level helper used by gpu_model_runner
# ---------------------------------------------------------------------------

def apply_tetris(
    draft_token_ids: torch.Tensor,
    draft_token_logprobs: torch.Tensor,
    num_speculative_tokens: int,
    extra_proposals: int = 0,
    turn_on_batch_size: Optional[int] = None,
) -> list[list[int]]:
    """Apply TETRIS to a batch of draft proposals.

    Converts a dense ``[batch_size, K]`` tensor of draft token IDs into a
    ragged ``list[list[int]]`` where each inner list has been trimmed to the
    TETRIS-optimal length for that request.

    The drafter generates ``K = num_speculative_tokens`` tokens per request
    (including ``extra_proposals`` additional positions).  The verification
    budget is ``base_k × batch_size`` where ``base_k = K − extra_proposals``.

    Args:
        draft_token_ids: Integer tensor ``[batch_size, K]``.
        draft_token_logprobs: Float tensor ``[batch_size, K]``
            of log-probabilities of the selected draft tokens at each step.
        num_speculative_tokens: Total draft length K produced by the drafter,
            equal to ``base_k + extra_proposals``.
        extra_proposals: Number of extra draft tokens beyond base_k.
            ``capacity = (K − extra_proposals) × B``.  Must be
            ``0 <= extra_proposals < num_speculative_tokens``.
        turn_on_batch_size: If set, TETRIS is only active when
            ``batch_size >= turn_on_batch_size``.

    Returns:
        A ragged list where ``result[i]`` is the list of draft token IDs for
        request *i*, truncated to the TETRIS-optimal length.
    """
    batch_size = draft_token_ids.shape[0]
    K = num_speculative_tokens
    base_k = K - extra_proposals

    if base_k <= 0:
        logger.warning_once(
            "TETRIS: extra_proposals (%d) >= num_speculative_tokens (%d); "
            "base_k would be %d. Falling back to full draft length.",
            extra_proposals,
            K,
            base_k,
            scope="local",
        )
        base_k = K

    # When TETRIS is inactive (batch too small), fall back to base_k tokens
    # (equivalent to vanilla with num_speculative_tokens = base_k).
    if turn_on_batch_size is not None and batch_size < turn_on_batch_size:
        logger.debug(
            "TETRIS: batch_size=%d < turn_on_batch_size=%d; "
            "falling back to base_k=%d tokens per request.",
            batch_size,
            turn_on_batch_size,
            base_k,
        )
        return [
            draft_token_ids[i, :base_k].tolist()
            for i in range(batch_size)
        ]

    # Paper formula: capacity = base_k × batch_size
    # The drafter produced K = base_k + extra tokens; TETRIS selects from
    # the wider K-token grid but caps verification at base_k × B slots.
    capacity = base_k * batch_size
    total_slots = K * batch_size

    if capacity >= total_slots:
        logger.info_once(
            "TETRIS: capacity (%d) >= total draft slots (%d). "
            "extra_proposals=%d is 0 or base_k >= K; selection is equivalent "
            "to vanilla. Set extra_proposals > 0 for TETRIS to have effect. "
            "Set %s=1 for per-step stats.",
            capacity,
            total_slots,
            extra_proposals,
            _TETRIS_DEBUG_ENV,
            scope="local",
        )

    num_draft_tokens_list = [K] * batch_size

    optimal_lengths = select_proposals(
        capacity=capacity,
        draft_token_logprobs=draft_token_logprobs,
        num_draft_tokens=num_draft_tokens_list,
    )

    result: list[list[int]] = []
    for i, length in enumerate(optimal_lengths):
        length = max(0, min(length, K))
        result.append(draft_token_ids[i, :length].tolist())

    total_after = sum(len(r) for r in result)
    logger.debug(
        "TETRIS: batch=%d K=%d base_k=%d extra=%d capacity=%d "
        "draft_tokens %d->%d (%.1f%% of full grid)",
        batch_size,
        K,
        base_k,
        extra_proposals,
        capacity,
        total_slots,
        total_after,
        100.0 * total_after / max(1, total_slots),
    )
    if _tetris_debug_enabled():
        logger.info(
            "TETRIS debug: batch=%d K=%d base_k=%d extra=%d "
            "capacity=%d grid=%d selected=%d "
            "per_request_lengths=%s",
            batch_size,
            K,
            base_k,
            extra_proposals,
            capacity,
            total_slots,
            total_after,
            optimal_lengths,
        )

    return result
