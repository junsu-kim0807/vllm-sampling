# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared runtime data structures for staged speculative decoding."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import torch

SpecStageMode = Literal[
    "draft_target",
    "inter_verification",
    "hierarchical_verification",
    "pivot",
]


@dataclass
class PivotExpansionFamily:
    """Expanded top-k candidate family for one origin request row."""

    origin_row: int
    expanded_rows: list[int]
    candidate_ranks: list[int]
    first_token_ids: list[int]
    first_token_probs: list[float]


@dataclass
class PivotExpansionPlan:
    """Runtime mapping between expanded pivot rows and origin rows."""

    expanded_to_origin: list[int]
    families: list[PivotExpansionFamily]
    expanded_batch_size: int


@dataclass(frozen=True)
class PivotTreeFamily:
    """One expanded root candidate that belongs to an origin request row."""

    origin_row: int
    family_id: int
    root_rank: int
    root_token_id: int
    # Flat verification span [start, end) for this family.
    node_row_start: int
    node_row_end: int


@dataclass(frozen=True)
class EagleTreeTemplate:
    """Shared Eagle tree topology used across expanded families."""

    parent_ids: list[int]
    node_depths: list[int]
    node_order: list[int]
    leaf_ids: list[int]
    num_nodes: int


@dataclass(frozen=True)
class PivotExpandedTreePlan:
    """Family-expanded plan + flat remap metadata for verifier contract."""

    families: list[PivotTreeFamily]
    template: EagleTreeTemplate
    # family-major mapping used by both intermediate and target verification.
    flat_to_family_ids: list[int]
    flat_to_node_ids: list[int]
    # [family_id] -> (start, end) in flat order.
    family_flat_spans: list[tuple[int, int]]
    expanded_to_origin: list[int]
    origin_batch_size: int


@dataclass
class FamilyTreeBundle:
    """Family-major tree tokens flattened for generic verifier contracts."""

    plan: PivotExpandedTreePlan
    # [num_flat_nodes]
    token_ids: torch.Tensor
    # [num_flat_nodes, vocab] or None.
    proposal_probs: torch.Tensor | None = None
    # Optional interpreted accepted path per family for iterative rounds.
    reduced_prefix_rows: list[list[int]] | None = None


@dataclass
class FamilyTreeReduceResult:
    """Interpreted verifier output reconstructed back to family semantics."""

    accepted_rows: list[list[int]]
    accepted_lens: list[int]
    chosen_leaf_ids: list[int]
    recovery_token_ids: list[int]


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
    # Optional pivot top-k expansion mapping.
    expansion_plan: PivotExpansionPlan | None = None
    # Optional tree-family flatten/reconstruction contract.
    tree_plan: PivotExpandedTreePlan | None = None


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
    # Optional expanded-row mapping.
    expansion_plan: PivotExpansionPlan | None = None
    # Optional tree-family flatten/reconstruction contract.
    tree_plan: PivotExpandedTreePlan | None = None


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


@dataclass(frozen=True)
class StagedHiddenStateBundle:
    """Hidden-state bundle owned by staged intermediate frontier."""

    hidden_states: torch.Tensor
    aux_hidden_states: torch.Tensor | None
    batch_size: int
    # Provisional KV frontier ownership marker.
    owns_provisional_frontier: bool = True


@dataclass
class IntermediateRoundState:
    """Mutable staged round state for intermediate frontier advancement."""

    hidden_bundle: StagedHiddenStateBundle | None = None
    verification_metadata: dict[str, object] | None = None
    # Copy-on-write block-table frontier metadata per expanded row.
    frontier_metadata: list[dict[str, object]] | None = None
    bootstrap_complete: bool = False


@dataclass(frozen=True)
class StagedVerificationResult:
    """Verification result with optional cleanup bookkeeping."""

    accepted_rows: list[list[int]]
    rejected_rows: list[list[int]]
    needs_cleanup_rows: list[int]


def _split_flat_tokens_by_lengths(
    flat: torch.Tensor, lengths: list[int]
) -> list[torch.Tensor]:
    rows: list[torch.Tensor] = []
    s = 0
    for l in lengths:
        li = int(l)
        if li > 0:
            rows.append(flat[s : s + li].clone())
            s += li
        else:
            rows.append(flat.new_empty((0,), dtype=flat.dtype, device=flat.device))
    return rows


def _split_probs_by_lengths(
    flat: torch.Tensor, lengths: list[int]
) -> list[torch.Tensor]:
    rows: list[torch.Tensor] = []
    s = 0
    for l in lengths:
        li = int(l)
        if li > 0:
            rows.append(flat[s : s + li].clone())
            s += li
        else:
            rows.append(
                flat.new_empty((0, flat.shape[-1]), dtype=flat.dtype, device=flat.device)
            )
    return rows


def expand_hybrid_bundle_for_pivot_expansion(
    bundle: HybridProposalBundle,
) -> HybridProposalBundle:
    """Duplicate per-origin draft rows into expanded_batch_size rows (pivot top-k).

    Each expanded row reuses the origin draft tail; the first pivot token is
    taken from :class:`PivotExpansionFamily` when applicable.
    """
    plan = bundle.expansion_plan
    if plan is None or plan.expanded_batch_size <= 0:
        return bundle
    if bundle.draft_probs is not None and plan.families:
        # Pivot top-k expansion currently reuses origin q(.) for multiple branches.
        # Disable stochastic proposal probs for expanded families until branch-
        # conditioned proposal math is implemented.
        bundle = replace(bundle, draft_probs=None)
    if len(plan.expanded_to_origin) != plan.expanded_batch_size:
        return bundle
    origin_b = len(bundle.num_draft_tokens)
    if origin_b == plan.expanded_batch_size:
        return bundle
    if origin_b == 0 or origin_b > plan.expanded_batch_size:
        return bundle

    tok_rows = _split_flat_tokens_by_lengths(
        bundle.draft_token_ids, bundle.num_draft_tokens
    )
    prob_rows: list[torch.Tensor] | None = None
    if bundle.draft_probs is not None and bundle.draft_probs.shape[0] == int(
        bundle.draft_token_ids.shape[0]
    ):
        prob_rows = _split_probs_by_lengths(bundle.draft_probs, bundle.num_draft_tokens)
    src_rows: list[torch.Tensor] | None = None
    if bundle.source_stage is not None and bundle.source_stage.shape[0] == int(
        bundle.draft_token_ids.shape[0]
    ):
        src_rows = _split_flat_tokens_by_lengths(
            bundle.source_stage, bundle.num_draft_tokens
        )

    new_lengths: list[int] = []
    new_tok: list[torch.Tensor] = []
    new_prob: list[torch.Tensor] = []
    new_src: list[torch.Tensor] = []

    for j, o in enumerate(plan.expanded_to_origin):
        if o < 0 or o >= origin_b:
            return bundle
        row_t = tok_rows[o].clone()
        for fam in plan.families:
            if j in fam.expanded_rows:
                li = fam.expanded_rows.index(j)
                row_t[0] = int(fam.first_token_ids[li])
                break
        new_lengths.append(int(row_t.shape[0]))
        new_tok.append(row_t)
        if prob_rows is not None:
            pr = prob_rows[o].clone()
            for fam in plan.families:
                if j in fam.expanded_rows:
                    li = fam.expanded_rows.index(j)
                    tid = int(fam.first_token_ids[li])
                    fp = float(fam.first_token_probs[li])
                    if pr.shape[0] > 0:
                        pr[0].zero_()
                        if 0 <= tid < pr.shape[-1]:
                            pr[0, tid] = fp
                    break
            new_prob.append(pr)
        if src_rows is not None:
            new_src.append(src_rows[o].clone())

    draft_token_ids = torch.cat(new_tok, dim=0).to(torch.int32)
    device = draft_token_ids.device
    cu = torch.cumsum(
        torch.tensor(new_lengths, dtype=torch.int32, device=device), dim=0
    )
    draft_probs = torch.cat(new_prob, dim=0) if new_prob else None
    source_stage = torch.cat(new_src, dim=0).to(torch.int32) if new_src else None
    max_spec_len = max(new_lengths) if new_lengths else bundle.max_spec_len

    return replace(
        bundle,
        draft_token_ids=draft_token_ids,
        draft_probs=draft_probs,
        num_draft_tokens=new_lengths,
        cu_num_draft_tokens=cu,
        max_spec_len=max_spec_len,
        source_stage=source_stage,
    )
