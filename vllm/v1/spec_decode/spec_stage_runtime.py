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
class RootTopKInfo:
    """Root draft step top-k tokens and probabilities (sorted descending per row).

    Used when pivot expansion needs q(·) at the first speculative position without
    materializing a full ``[B, 1, vocab]`` tensor (e.g. parallel drafting).
    ``topk_probs`` are softmax probabilities from the same logits as ``topk_token_ids``.
    """

    topk_token_ids: torch.Tensor  # [B, K]
    topk_probs: torch.Tensor  # [B, K], largest first


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
    """Runtime mapping between expanded pivot rows and origin rows.

    Fixed-capacity packing (linear pivot, PR1): ``packed_to_origin`` may contain
    ``-1`` for inactive extra slots; **never** index Python lists with
    ``packed_to_origin[j]``. Use ``packed_sm_origin[j]`` (always in ``[0, B)``).

    ``expanded_to_origin`` is a **separate** list with the same values as
    ``packed_sm_origin`` (never an alias of ``packed_to_origin``).
    """

    expanded_to_origin: list[int]
    families: list[PivotExpansionFamily]
    expanded_batch_size: int
    # --- Fixed-capacity packed layout (linear pivot); eagle-tree legacy may omit. ---
    origin_batch_size: int = 0
    """Origin batch B; 0 means legacy variable expansion (eagle tree path)."""
    packed_batch_size: int = 0
    """Equals expanded_batch_size when packing is used."""
    packed_to_origin: list[int] | None = None
    """Length P; active rows in ``[0, B)``; inactive extra slots ``-1``."""
    packed_sm_origin: list[int] | None = None
    """Length P; list-index / cu_meta anchor; every entry in ``[0, B)``."""
    packed_row_is_active: list[bool] | None = None
    packed_row_is_base: list[bool] | None = None
    packed_row_family_rank: list[int] | None = None
    origin_to_base_row: list[int] | None = None
    origin_to_family_rows: list[list[int]] | None = None
    uses_fixed_capacity_packing: bool = False
    """True for linear pivot fixed P layout; false for legacy variable eagle expansion."""
    # Optional packed layout tensors (device = plan construction device); list fields
    # above remain the compatibility surface for consumers/tests.
    packed_to_origin_t: torch.Tensor | None = None
    """Shape ``[P]``; inactive slots ``-1`` (same semantics as ``packed_to_origin``)."""
    packed_sm_origin_t: torch.Tensor | None = None
    """Shape ``[P]``; indices in ``[0, B)`` for sampling-metadata / gather."""
    packed_row_is_active_t: torch.Tensor | None = None
    """Shape ``[P]``, bool."""
    packed_row_family_rank_t: torch.Tensor | None = None
    """Shape ``[P]``; inactive rows ``-1``."""
    row_gather_idx_cpu: torch.Tensor | None = None
    """Long tensor ``[P]`` on CPU; equals ``packed_sm_origin`` row-major gather indices."""
    # --- Tensor-only expansion metadata (linear fixed P; ``families`` may be empty) ---
    packed_row_is_base_t: torch.Tensor | None = None
    """Shape ``[P]``; bool packed as 0/1 int8 or bool tensor (same semantics as list)."""
    family_origin_rows_t: torch.Tensor | None = None
    """Shape ``[F]`` long; origin row index per expansion family."""
    family_expanded_rows_t: torch.Tensor | None = None
    """Shape ``[F, R]`` long; packed row indices per family, ``-1`` padding."""
    family_candidate_ranks_t: torch.Tensor | None = None
    """Shape ``[F, R]`` int32; tie-break rank aligned with ``family_expanded_rows_t``."""
    family_first_token_ids_t: torch.Tensor | None = None
    """Shape ``[F, R]`` int32; first-token id per expanded row slot."""
    family_first_token_probs_t: torch.Tensor | None = None
    """Shape ``[F, R]`` float32; draft prob per expanded row slot."""
    origin_to_family_rows_t: torch.Tensor | None = None
    """Shape ``[B, M]`` long; extra packed row indices per origin, ``-1`` padding."""
    origin_to_family_rows_count_t: torch.Tensor | None = None
    """Shape ``[B]`` int32; valid entries per row in ``origin_to_family_rows_t``."""
    origin_to_base_row_t: torch.Tensor | None = None
    """Shape ``[B]`` long; base packed row index for each origin."""


def pivot_plan_sm_origin_cpu_long(plan: PivotExpansionPlan) -> torch.Tensor:
    """Packed row ``j`` -> scheduler origin in ``[0, B)`` (CPU long ``[P]``)."""
    if plan.row_gather_idx_cpu is not None and int(plan.row_gather_idx_cpu.shape[0]) == int(
        plan.expanded_batch_size
    ):
        return plan.row_gather_idx_cpu.detach().to(torch.long).contiguous()
    if plan.packed_sm_origin_t is not None:
        return plan.packed_sm_origin_t.detach().cpu().to(torch.long).contiguous()
    sm = plan.packed_sm_origin if plan.packed_sm_origin is not None else plan.expanded_to_origin
    return torch.tensor(sm, dtype=torch.long)


def pivot_plan_origin_to_base_row_cpu_long(plan: PivotExpansionPlan) -> torch.Tensor:
    """Origin ``o`` in ``[0, B)`` -> packed row index of that origin's **base** row (CPU long ``[B]``).

    Prefer explicit ``origin_to_base_row_t`` / ``origin_to_base_row``; otherwise derive
    the first packed row per origin from ``pivot_plan_sm_origin_cpu_long`` (tensor-only
    safe; list ``expanded_to_origin`` may be empty).
    """
    b = int(plan.origin_batch_size)
    if b <= 0:
        sm0 = pivot_plan_sm_origin_cpu_long(plan)
        if sm0.numel() == 0:
            return torch.empty(0, dtype=torch.long)
        b = int(sm0.max().item()) + 1

    if plan.origin_to_base_row_t is not None and int(plan.origin_to_base_row_t.shape[0]) >= b:
        return plan.origin_to_base_row_t[:b].detach().cpu().to(torch.long).contiguous()

    if plan.origin_to_base_row is not None and len(plan.origin_to_base_row) >= b:
        return torch.tensor(plan.origin_to_base_row[:b], dtype=torch.long)

    sm = pivot_plan_sm_origin_cpu_long(plan)
    out = torch.full((b,), -1, dtype=torch.long)
    for j in range(int(sm.shape[0])):
        o = int(sm[j].item())
        if 0 <= o < b and int(out[o].item()) < 0:
            out[o] = j
    return out


def pivot_plan_uses_tensor_only_family_meta(plan: PivotExpansionPlan) -> bool:
    return bool(
        plan.uses_fixed_capacity_packing
        and plan.family_expanded_rows_t is not None
        and (not plan.families)
    )


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
    # Per bundle row: scheduler request id at propose time (same length as
    # ``num_draft_tokens``). Used to reorder / validate against the current
    # batch when prepare-time metadata rows differ from propose-time order.
    bundle_row_req_ids: tuple[str, ...] | None = None
    # Optional per-row staged-verification totals recorded before target verify.
    # May be a 1-D CPU int tensor (e.g. HV loop) or a Python list at boundaries.
    inter_verified_counts: torch.Tensor | list[int] | None = None
    inter_accepted_counts: torch.Tensor | list[int] | None = None
    staged_verification_depth: int = 0


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
    # When present, ``hidden_states`` rows align with ``pref_cad`` (already
    # prefix-conditioned). Tuple is
    # ``(pref_toks, pref_pos, pref_hidden, pref_next, pref_cad)`` matching
    # ``_build_prefix_conditioned_inputs`` output — use for follow-on propose /
    # verify without re-slicing against the *base* attention metadata.
    prefix_prefab: tuple | None = None


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


def pivot_expansion_indices_fit_prepare_batch(
    plan: PivotExpansionPlan,
    num_reqs: int,
    *,
    expected_packed_size: int | None = None,
) -> bool:
    """Validate pivot plan against the current scheduler batch and optional P formula."""
    if num_reqs <= 0:
        return False
    if plan.expanded_batch_size <= 0:
        return False
    p = int(plan.expanded_batch_size)
    if len(plan.expanded_to_origin) != p:
        if plan.packed_sm_origin_t is None or int(plan.packed_sm_origin_t.shape[0]) != p:
            return False
    # Fixed-capacity packed layout (list-backed packed_sm_origin)
    if plan.packed_sm_origin is not None:
        if plan.packed_batch_size != p or len(plan.packed_sm_origin) != p:
            return False
        if plan.origin_batch_size != num_reqs:
            return False
        if (
            plan.uses_fixed_capacity_packing
            and expected_packed_size is not None
            and p != expected_packed_size
        ):
            return False
        for x in plan.packed_sm_origin:
            if int(x) < 0 or int(x) >= num_reqs:
                return False
        if plan.packed_to_origin is not None:
            if len(plan.packed_to_origin) != p:
                return False
            if plan.packed_row_is_active is None or len(plan.packed_row_is_active) != p:
                return False
            for j in range(p):
                active = bool(plan.packed_row_is_active[j])
                po = int(plan.packed_to_origin[j])
                if active and (po < 0 or po >= num_reqs):
                    return False
                if not active and po != -1:
                    return False
        if plan.uses_fixed_capacity_packing:
            # PR1 linear packed invariants (see PivotExpansionPlan / pivot builder).
            if plan.packed_to_origin is None or plan.packed_row_is_active is None:
                return False
            sm = plan.packed_sm_origin
            if sm is None:
                return False
            if plan.origin_to_base_row is None or len(plan.origin_to_base_row) != num_reqs:
                return False
            for o in range(num_reqs):
                if int(plan.origin_to_base_row[o]) != o:
                    return False
            if plan.packed_row_is_base is None or len(plan.packed_row_is_base) != p:
                return False
            if sum(1 for b in plan.packed_row_is_base if b) != num_reqs:
                return False
            for o in range(num_reqs):
                if not plan.packed_row_is_base[o]:
                    return False
            for j in range(num_reqs, p):
                if plan.packed_row_is_base[j]:
                    return False
            for o in range(num_reqs):
                if not bool(plan.packed_row_is_active[o]):
                    return False
                if int(plan.packed_to_origin[o]) != o:
                    return False
                if int(sm[o]) != o:
                    return False
            if len(plan.expanded_to_origin) == p and any(
                int(plan.expanded_to_origin[j]) != int(sm[j]) for j in range(p)
            ):
                return False
        return True
    # Tensor-only packed layout (linear fixed P; no list ``packed_sm_origin``)
    if (
        plan.packed_sm_origin_t is not None
        and plan.packed_to_origin_t is not None
        and plan.packed_row_is_active_t is not None
        and plan.packed_row_is_base_t is not None
    ):
        if plan.packed_batch_size != p or int(plan.packed_sm_origin_t.shape[0]) != p:
            return False
        if plan.origin_batch_size != num_reqs:
            return False
        if (
            plan.uses_fixed_capacity_packing
            and expected_packed_size is not None
            and p != expected_packed_size
        ):
            return False
        sm = plan.packed_sm_origin_t
        if bool((sm < 0).any().item()) or bool((sm >= num_reqs).any().item()):
            return False
        pto = plan.packed_to_origin_t
        act = plan.packed_row_is_active_t
        isb = plan.packed_row_is_base_t
        if int(pto.shape[0]) != p or int(act.shape[0]) != p or int(isb.shape[0]) != p:
            return False
        for j in range(p):
            active = bool(act[j].item())
            po = int(pto[j].item())
            if active and (po < 0 or po >= num_reqs):
                return False
            if not active and po != -1:
                return False
        if plan.uses_fixed_capacity_packing:
            otb = plan.origin_to_base_row_t
            if otb is None or int(otb.shape[0]) != num_reqs:
                return False
            ar = torch.arange(num_reqs, device=otb.device, dtype=otb.dtype)
            if not torch.equal(otb, ar):
                return False
            if int(isb[:num_reqs].sum().item()) != num_reqs:
                return False
            for o in range(num_reqs):
                if not bool(isb[o].item()):
                    return False
            for j in range(num_reqs, p):
                if bool(isb[j].item()):
                    return False
            for o in range(num_reqs):
                if not bool(act[o].item()):
                    return False
                if int(pto[o].item()) != o:
                    return False
                if int(sm[o].item()) != o:
                    return False
            if len(plan.expanded_to_origin) == p and any(
                int(plan.expanded_to_origin[j]) != int(sm[j].item()) for j in range(p)
            ):
                return False
        return True
    # Legacy dense expansion (e.g. eagle tree): expanded_to_origin is all valid origins
    if not plan.expanded_to_origin:
        return False
    for o in plan.expanded_to_origin:
        if int(o) < 0 or int(o) >= num_reqs:
            return False
    return True


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


def _pivot_row_first_token_id_from_family_tensors(
    plan: PivotExpansionPlan, j: int
) -> int | None:
    """First speculative token id for packed row ``j`` from tensor family tables."""
    if plan.families:
        return None
    fer = plan.family_expanded_rows_t
    ftok = plan.family_first_token_ids_t
    if fer is None or ftok is None:
        return None
    mask = fer == int(j)
    if not bool(mask.any().item()):
        return None
    fr = torch.nonzero(mask, as_tuple=False)[0]
    return int(ftok[int(fr[0].item()), int(fr[1].item())].item())


def expand_hybrid_bundle_for_pivot_expansion(
    bundle: HybridProposalBundle,
) -> HybridProposalBundle:
    """Apply pivot expansion to the hybrid bundle (densify B->P or sync packed rows).

    Fixed-capacity linear pivot: the proposer already emits ``P`` rows; inactive
    rows must carry zero draft width. Legacy eagle-tree path still densifies
    from origin batch ``B`` to variable ``B'``.
    """
    plan = bundle.expansion_plan
    if plan is None or plan.expanded_batch_size <= 0:
        return bundle
    if (
        bundle.draft_probs is not None
        and plan.families
        and not plan.uses_fixed_capacity_packing
    ):
        # Legacy variable expansion: origin q(.) is shared across branches.
        # Fixed-capacity path: ``expand_hybrid_bundle`` clones rows and patches
        # first-token mass from ``PivotExpansionFamily.first_token_probs`` (PR3).
        bundle = replace(bundle, draft_probs=None)
    p_sz = int(plan.expanded_batch_size)
    if len(plan.expanded_to_origin) != p_sz:
        if plan.packed_sm_origin_t is None or int(plan.packed_sm_origin_t.shape[0]) != p_sz:
            return bundle

    # Already packed: enforce inactive rows have zero draft tokens.
    if (
        plan.packed_sm_origin_t is not None
        and plan.packed_row_is_active_t is not None
        and len(bundle.num_draft_tokens) == plan.packed_batch_size
        and int(plan.packed_sm_origin_t.shape[0]) == plan.packed_batch_size
    ):
        p = plan.packed_batch_size
        act = plan.packed_row_is_active_t
        lengths = list(bundle.num_draft_tokens)
        needs_rebuild = any(
            (not bool(act[j].item())) and lengths[j] != 0 for j in range(p)
        )
        if not needs_rebuild:
            return bundle
        tok_rows = _split_flat_tokens_by_lengths(
            bundle.draft_token_ids, bundle.num_draft_tokens
        )
        new_lengths: list[int] = []
        new_tok: list[torch.Tensor] = []
        for j in range(p):
            if bool(act[j].item()):
                new_lengths.append(lengths[j])
                new_tok.append(tok_rows[j])
            else:
                new_lengths.append(0)
                new_tok.append(
                    tok_rows[j].new_empty((0,), dtype=tok_rows[j].dtype)
                )
        draft_token_ids = (
            torch.cat(new_tok, dim=0).to(torch.int32)
            if any(l > 0 for l in new_lengths)
            else bundle.draft_token_ids.new_empty((0,), dtype=torch.int32)
        )
        device = draft_token_ids.device
        cu = torch.cumsum(
            torch.tensor(new_lengths, dtype=torch.int32, device=device), dim=0
        ).to(torch.int32)
        max_spec_len = max(new_lengths) if any(new_lengths) else bundle.max_spec_len
        return replace(
            bundle,
            draft_token_ids=draft_token_ids,
            draft_probs=None,
            num_draft_tokens=new_lengths,
            cu_num_draft_tokens=cu,
            max_spec_len=max_spec_len,
            source_stage=None,
        )

    if (
        plan.packed_sm_origin is not None
        and plan.packed_row_is_active is not None
        and len(bundle.num_draft_tokens) == plan.packed_batch_size
    ):
        p = plan.packed_batch_size
        lengths = list(bundle.num_draft_tokens)
        needs_rebuild = any(
            (not plan.packed_row_is_active[j]) and lengths[j] != 0
            for j in range(p)
        )
        if not needs_rebuild:
            return bundle
        tok_rows = _split_flat_tokens_by_lengths(
            bundle.draft_token_ids, bundle.num_draft_tokens
        )
        new_lengths: list[int] = []
        new_tok: list[torch.Tensor] = []
        for j in range(p):
            if plan.packed_row_is_active[j]:
                new_lengths.append(lengths[j])
                new_tok.append(tok_rows[j])
            else:
                new_lengths.append(0)
                new_tok.append(
                    tok_rows[j].new_empty((0,), dtype=tok_rows[j].dtype)
                )
        draft_token_ids = (
            torch.cat(new_tok, dim=0).to(torch.int32)
            if any(l > 0 for l in new_lengths)
            else bundle.draft_token_ids.new_empty((0,), dtype=torch.int32)
        )
        device = draft_token_ids.device
        cu = torch.cumsum(
            torch.tensor(new_lengths, dtype=torch.int32, device=device), dim=0
        ).to(torch.int32)
        max_spec_len = max(new_lengths) if any(new_lengths) else bundle.max_spec_len
        return replace(
            bundle,
            draft_token_ids=draft_token_ids,
            draft_probs=None,
            num_draft_tokens=new_lengths,
            cu_num_draft_tokens=cu,
            max_spec_len=max_spec_len,
            source_stage=None,
        )

    origin_b = len(bundle.num_draft_tokens)
    if origin_b == plan.expanded_batch_size:
        return bundle
    if origin_b == 0 or origin_b > plan.expanded_batch_size:
        return bundle

    sm = plan.packed_sm_origin
    if sm is not None and len(sm) == plan.expanded_batch_size:
        sm_list = sm
    elif (
        plan.packed_sm_origin_t is not None
        and int(plan.packed_sm_origin_t.shape[0]) == plan.expanded_batch_size
    ):
        sm_list = plan.packed_sm_origin_t.detach().cpu().tolist()
    else:
        sm_list = plan.expanded_to_origin

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

    for j, o in enumerate(sm_list):
        if o < 0 or o >= origin_b:
            return bundle
        row_t = tok_rows[o].clone()
        patch_id = _pivot_row_first_token_id_from_family_tensors(plan, j)
        if patch_id is not None:
            row_t[0] = patch_id
        else:
            for fam in plan.families:
                if j in fam.expanded_rows:
                    li = fam.expanded_rows.index(j)
                    row_t[0] = int(fam.first_token_ids[li])
                    break
        new_lengths.append(int(row_t.shape[0]))
        new_tok.append(row_t)
        if prob_rows is not None:
            pr = prob_rows[o].clone()
            prob_patched = False
            if patch_id is not None and plan.family_first_token_probs_t is not None:
                fer = plan.family_expanded_rows_t
                fpr = plan.family_first_token_probs_t
                if fer is not None:
                    mask = fer == int(j)
                    if bool(mask.any().item()):
                        fr = torch.nonzero(mask, as_tuple=False)[0]
                        fp = float(
                            fpr[int(fr[0].item()), int(fr[1].item())].item()
                        )
                        tid = int(patch_id)
                        if pr.shape[0] > 0:
                            pr[0].zero_()
                            if 0 <= tid < pr.shape[-1]:
                                pr[0, tid] = fp
                        prob_patched = True
            if not prob_patched:
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
    ).to(torch.int32)
    draft_probs = torch.cat(new_prob, dim=0) if new_prob else None
    source_stage = torch.cat(new_src, dim=0).to(torch.int32) if new_src else None
    max_spec_len = max(new_lengths) if new_lengths else bundle.max_spec_len

    new_br: tuple[str, ...] | None = None
    br = bundle.bundle_row_req_ids
    if br is not None and len(br) == origin_b:
        new_br = tuple(str(br[int(sm_list[j])]) for j in range(len(sm_list)))

    return replace(
        bundle,
        draft_token_ids=draft_token_ids,
        draft_probs=draft_probs,
        num_draft_tokens=new_lengths,
        cu_num_draft_tokens=cu,
        max_spec_len=max_spec_len,
        source_stage=source_stage,
        bundle_row_req_ids=new_br,
    )
