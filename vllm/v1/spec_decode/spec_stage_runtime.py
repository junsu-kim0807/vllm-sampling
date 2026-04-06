# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared runtime data structures for staged speculative decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

SpecStageMode = Literal[
    "draft_target",
    "inter_verification",
    "hierarchical_verification",
    "pivot",
]


@dataclass
class HybridProposalBundle:
    """Proposal payload passed from proposer/controller to target verification."""

    # Flattened in cu-segment order.
    draft_token_ids: torch.Tensor
    # [num_tokens, vocab_size] if available.
    draft_probs: torch.Tensor | None
    # Per-request proposal lengths.
    num_draft_tokens: list[int]
    # Inclusive prefix sums of num_draft_tokens.
    cu_num_draft_tokens: torch.Tensor
    max_spec_len: int
    mode: SpecStageMode
    # Optional stage tag per token: 0 = I-equivalent prefix, 1 = D tail.
    source_stage: torch.Tensor | None = None


@dataclass
class VerificationRows:
    """Parsed verification outputs in request-major form."""

    emitted_rows: list[list[int]]
    emitted_probs_rows: list[torch.Tensor] | None
    accepted_lens: list[int]
    had_bonus: list[bool]


@dataclass
class RequestAdaptiveState:
    """Per-request adaptive-stage selection state."""

    current_mode: SpecStageMode = "draft_target"
    cooldown: int = 0
    steps_seen: int = 0
    ema_goodput_dt: float = 0.0
    ema_goodput_it: float = 0.0
    ema_goodput_dit: float = 0.0
    ema_accept_dt: float = 0.0
    ema_accept_it: float = 0.0
    ema_accept_di: float = 0.0
    ema_prop_us_dt: float = 0.0
    ema_prop_us_it: float = 0.0
    ema_prop_us_dit: float = 0.0


@dataclass
class DitPrefixState:
    """Logical prefix accepted by intermediate verification in current step."""

    # [B, P_max]
    tokens: torch.Tensor
    # [B]
    lengths: torch.Tensor
    # Optional draft probs aligned to tokens [sum(lengths), vocab]
    probs_flat: torch.Tensor | None = None
    # Optional provenance flattened to valid tokens.
    source_stage_flat: torch.Tensor | None = None


@dataclass
class DitRoundProposal:
    """Per-round draft proposal (chunk-level)."""

    # [B, L]
    tokens: torch.Tensor
    # [B, L, vocab] or None
    probs: torch.Tensor | None = None


@dataclass
class DitRoundVerification:
    """Intermediate verification payload for one chunk."""

    # [B*L, vocab]
    logits_flat: torch.Tensor
    # [B, vocab]
    bonus_logits: torch.Tensor


@dataclass
class DitRoundDecision:
    """Intermediate accept/reject output for one chunk."""

    # variable-length emitted rows per request
    emitted_rows: list[list[int]]
    emitted_prob_rows: list[list[torch.Tensor]]
    accepted_lens: list[int]


@dataclass
class DitAssembledBundleState:
    """Assembled candidate tokens to pass to final target verification."""

    # [B, cap] with PLACEHOLDER_TOKEN_ID padding.
    tokens: torch.Tensor
    # flattened [num_valid, vocab] if available.
    probs_flat: torch.Tensor | None
    # Optional per-row stage provenance.
    source_stage_rows: list[list[int]]


@dataclass
class DitRoundState:
    """Token-buffer-only round state for one outer DIT iteration."""

    req_ids: list[str]
    # [B]
    base_sampled_token_ids: torch.Tensor
    prefix_rows: list[list[int]]
    prefix_prob_rows: list[list[torch.Tensor]]
    prefix_source_rows: list[list[int]]
