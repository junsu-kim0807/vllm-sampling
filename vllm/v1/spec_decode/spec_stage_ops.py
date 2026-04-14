# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generic stage operations reused by adaptive/staged speculative decoding."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, replace

import torch

from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p, random_sample
from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID, RejectionSampler
from vllm.v1.sample.sampler import Sampler, _SAMPLING_EPS
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.spec_stage_runtime import (
    EagleTreeTemplate,
    FamilyTreeReduceResult,
    IntermediateRoundState,
    HybridProposalBundle,
    PivotExpandedTreePlan,
    PivotExpansionFamily,
    PivotExpansionPlan,
    PivotTreeFamily,
    _split_flat_tokens_by_lengths,
    _split_probs_by_lengths,
)


def _pivot_packed_row_is_active(plan: PivotExpansionPlan, row_idx: int) -> bool:
    """Whether ``row_idx`` participates in family competition / cleanup (linear packed)."""
    active = plan.packed_row_is_active
    if active is None:
        return True
    if row_idx < 0 or row_idx >= len(active):
        return False
    return bool(active[row_idx])


def get_accepted_draft_lens_from_sampled_tokens(
    sampled_token_ids: torch.Tensor,
    *,
    placeholder_token_id: int,
) -> list[int]:
    """Count non-placeholder entries per row (emitted width; may include bonus slot)."""
    return [
        int((row != placeholder_token_id).sum().item())
        for row in sampled_token_ids
    ]


def get_target_verification_accepted_draft_prefix_lens(
    sampled_token_ids: torch.Tensor,
    num_draft_tokens: list[int],
    *,
    placeholder_token_id: int,
) -> list[int]:
    """Accepted draft-prefix length per row, ignoring bonus/recovery past num_draft."""
    out: list[int] = []
    for r, n_draft in enumerate(num_draft_tokens):
        row = sampled_token_ids[r]
        cnt = 0
        for j in range(min(int(n_draft), int(row.shape[0]))):
            if int(row[j].item()) == placeholder_token_id:
                break
            cnt += 1
        out.append(cnt)
    return out


def build_family_flatten_order(
    *,
    families: list[PivotTreeFamily],
    template: EagleTreeTemplate,
    origin_batch_size: int,
) -> PivotExpandedTreePlan:
    """Build family-major flat ordering metadata (single source of truth)."""
    sorted_families = sorted(families, key=lambda fam: fam.family_id)
    flat_to_family_ids: list[int] = []
    flat_to_node_ids: list[int] = []
    family_flat_spans: list[tuple[int, int]] = []
    expanded_to_origin: list[int] = []
    start = 0
    for fam in sorted_families:
        end = start + int(template.num_nodes)
        family_flat_spans.append((start, end))
        expanded_to_origin.append(int(fam.origin_row))
        flat_to_family_ids.extend([int(fam.family_id)] * int(template.num_nodes))
        flat_to_node_ids.extend(list(template.node_order))
        start = end
    plan = PivotExpandedTreePlan(
        families=sorted_families,
        template=template,
        flat_to_family_ids=flat_to_family_ids,
        flat_to_node_ids=flat_to_node_ids,
        family_flat_spans=family_flat_spans,
        expanded_to_origin=expanded_to_origin,
        origin_batch_size=int(origin_batch_size),
    )
    for fam_idx, (s, e) in enumerate(plan.family_flat_spans):
        assert int(e - s) == int(plan.template.num_nodes), (
            "pivot_tree_family_span_width "
            f"fam={fam_idx}, span=({s},{e}), num_nodes={plan.template.num_nodes}"
        )
        assert all(fid == fam_idx for fid in plan.flat_to_family_ids[s:e]), (
            "pivot_tree_family_major_ids "
            f"fam={fam_idx}"
        )
        assert plan.flat_to_node_ids[s:e] == plan.template.node_order, (
            "pivot_tree_node_order_match "
            f"fam={fam_idx}"
        )
    return plan


def _is_ancestor_chain_accepted(
    node_id: int,
    *,
    accepted_mask: list[bool],
    template: EagleTreeTemplate,
) -> bool:
    cur = int(node_id)
    while cur >= 0:
        if cur >= len(accepted_mask) or not accepted_mask[cur]:
            return False
        if cur >= len(template.parent_ids):
            return False
        cur = int(template.parent_ids[cur])
    return True


def _infer_chosen_leaf_id(
    *,
    accepted_mask: list[bool],
    template: EagleTreeTemplate,
) -> int:
    chosen_leaf = -1
    chosen_depth = -1
    for leaf_id in template.leaf_ids:
        leaf = int(leaf_id)
        if not _is_ancestor_chain_accepted(
            leaf, accepted_mask=accepted_mask, template=template
        ):
            continue
        depth = (
            int(template.node_depths[leaf])
            if 0 <= leaf < len(template.node_depths)
            else -1
        )
        if depth > chosen_depth:
            chosen_leaf = leaf
            chosen_depth = depth
    if chosen_leaf >= 0:
        return chosen_leaf
    # Fallback: deepest accepted node that has accepted ancestor chain.
    best = -1
    best_depth = -1
    for node_id, is_acc in enumerate(accepted_mask):
        if not is_acc:
            continue
        if not _is_ancestor_chain_accepted(
            node_id, accepted_mask=accepted_mask, template=template
        ):
            continue
        depth = (
            int(template.node_depths[node_id])
            if node_id < len(template.node_depths)
            else -1
        )
        if depth > best_depth:
            best = int(node_id)
            best_depth = depth
    return best


def _reconstruct_node_path(
    *,
    leaf_id: int,
    template: EagleTreeTemplate,
    accepted_mask: list[bool],
) -> list[int]:
    if leaf_id < 0:
        return []
    path_rev: list[int] = []
    cur = int(leaf_id)
    while cur >= 0:
        if cur >= len(accepted_mask) or not accepted_mask[cur]:
            break
        path_rev.append(cur)
        if cur >= len(template.parent_ids):
            break
        cur = int(template.parent_ids[cur])
    return list(reversed(path_rev))


def collapse_family_tree_sampled_to_family_paths(
    sampled_token_ids: torch.Tensor,
    *,
    plan: PivotExpandedTreePlan,
    num_draft_tokens: list[int] | None = None,
) -> FamilyTreeReduceResult:
    """Interpret flat verifier outputs back into per-family tree-path semantics."""
    num_families = len(plan.families)
    if num_families == 0:
        return FamilyTreeReduceResult(
            accepted_rows=[],
            accepted_lens=[],
            chosen_leaf_ids=[],
            recovery_token_ids=[],
        )
    if sampled_token_ids.shape[0] < num_families:
        return FamilyTreeReduceResult(
            accepted_rows=[],
            accepted_lens=[],
            chosen_leaf_ids=[],
            recovery_token_ids=[],
        )
    if num_draft_tokens is None or len(num_draft_tokens) != num_families:
        num_draft_tokens = [int(plan.template.num_nodes)] * num_families
    accepted_lens = get_target_verification_accepted_draft_prefix_lens(
        sampled_token_ids[:num_families],
        num_draft_tokens,
        placeholder_token_id=PLACEHOLDER_TOKEN_ID,
    )
    accepted_rows: list[list[int]] = []
    chosen_leaf_ids: list[int] = []
    recovery_token_ids: list[int] = []
    reconstructed_lens: list[int] = []
    for fam_idx in range(num_families):
        row = sampled_token_ids[fam_idx]
        accepted_len_raw = int(accepted_lens[fam_idx])
        span_start, span_end = plan.family_flat_spans[fam_idx]
        node_order = plan.flat_to_node_ids[span_start:span_end]
        node_to_pos = {int(node_id): pos for pos, node_id in enumerate(node_order)}
        accepted_mask = [False] * int(plan.template.num_nodes)
        max_local = min(accepted_len_raw, len(node_order))
        for local_pos in range(max_local):
            node_id = int(node_order[local_pos])
            if 0 <= node_id < len(accepted_mask):
                accepted_mask[node_id] = True
        chosen_leaf = _infer_chosen_leaf_id(
            accepted_mask=accepted_mask,
            template=plan.template,
        )
        chosen_leaf_ids.append(chosen_leaf)
        path_node_ids = _reconstruct_node_path(
            leaf_id=chosen_leaf,
            template=plan.template,
            accepted_mask=accepted_mask,
        )
        path_tokens: list[int] = []
        for node_id in path_node_ids:
            if node_id not in node_to_pos:
                continue
            pos = int(node_to_pos[node_id])
            if pos < row.shape[0]:
                tok = int(row[pos].item())
                if tok != PLACEHOLDER_TOKEN_ID:
                    path_tokens.append(tok)
        accepted_rows.append(path_tokens)
        reconstructed_lens.append(len(path_tokens))
        assert reconstructed_lens[fam_idx] <= accepted_len_raw, (
            "pivot_tree_reconstructed_len_le_raw "
            f"fam={fam_idx}, raw={accepted_len_raw}, recon={reconstructed_lens[fam_idx]}"
        )
        assert all(tok != PLACEHOLDER_TOKEN_ID for tok in accepted_rows[fam_idx]), (
            "pivot_tree_reconstructed_no_placeholder "
            f"fam={fam_idx}"
        )
        assert chosen_leaf_ids[fam_idx] < 0 or len(path_node_ids) == reconstructed_lens[fam_idx], (
            "pivot_tree_path_token_count_match "
            f"fam={fam_idx}, leaf={chosen_leaf_ids[fam_idx]}"
        )
        recovery_idx = min(accepted_len_raw, int(row.shape[0]) - 1)
        rec = int(row[recovery_idx].item()) if recovery_idx >= 0 else PLACEHOLDER_TOKEN_ID
        recovery_token_ids.append(rec)
    return FamilyTreeReduceResult(
        accepted_rows=accepted_rows,
        accepted_lens=reconstructed_lens,
        chosen_leaf_ids=chosen_leaf_ids,
        recovery_token_ids=recovery_token_ids,
    )


def collapse_family_paths_to_origin(
    sampled_token_ids: torch.Tensor,
    *,
    plan: PivotExpandedTreePlan,
    reduced: FamilyTreeReduceResult,
) -> tuple[torch.Tensor, list[int]]:
    """Collapse families to origin rows by max accepted length then root rank."""
    if sampled_token_ids.shape[0] < len(plan.families):
        return sampled_token_ids, list(range(int(sampled_token_ids.shape[0])))
    selected_family_rows: list[int] = []
    by_origin: dict[int, list[PivotTreeFamily]] = {}
    for fam in plan.families:
        by_origin.setdefault(int(fam.origin_row), []).append(fam)
    for origin_row in range(int(plan.origin_batch_size)):
        fams = by_origin.get(origin_row, [])
        if not fams:
            fallback_row = next(
                (idx for idx, org in enumerate(plan.expanded_to_origin) if org == origin_row),
                0,
            )
            selected_family_rows.append(int(fallback_row))
            continue
        # Max accepted length, then lower root rank.
        best = min(
            fams,
            key=lambda fam: (
                -int(reduced.accepted_lens[fam.family_id]),
                int(fam.root_rank),
                int(fam.family_id),
            ),
        )
        selected_family_rows.append(int(best.family_id))
    device = sampled_token_ids.device
    idx = torch.tensor(selected_family_rows, device=device, dtype=torch.long)
    return sampled_token_ids.index_select(0, idx), selected_family_rows


def select_pivot_expanded_rows_to_origin(
    sampled_token_ids: torch.Tensor,
    expansion_plan: PivotExpansionPlan,
    *,
    num_draft_tokens: list[int] | None = None,
) -> list[int]:
    """Select one expanded row per origin using collapse tie-break rules.

    Tie-break order:
    1) larger accepted prefix length
    2) larger first-token probability
    3) smaller candidate rank
    """
    if sampled_token_ids.shape[0] < expansion_plan.expanded_batch_size:
        return list(range(int(sampled_token_ids.shape[0])))
    if not expansion_plan.expanded_to_origin:
        return list(range(int(sampled_token_ids.shape[0])))
    if (
        num_draft_tokens is not None
        and len(num_draft_tokens) == expansion_plan.expanded_batch_size
    ):
        accepted_lens = get_target_verification_accepted_draft_prefix_lens(
            sampled_token_ids,
            num_draft_tokens,
            placeholder_token_id=PLACEHOLDER_TOKEN_ID,
        )
    else:
        accepted_lens = get_accepted_draft_lens_from_sampled_tokens(
            sampled_token_ids,
            placeholder_token_id=PLACEHOLDER_TOKEN_ID,
        )
    b_origin = (
        int(expansion_plan.origin_batch_size)
        if expansion_plan.origin_batch_size > 0
        else max(expansion_plan.expanded_to_origin) + 1
    )

    # Fixed-capacity fast path: build [B, W] candidate table and perform tie-break
    # with tensor ops. Keep general Python loop fallback below.
    if (
        expansion_plan.uses_fixed_capacity_packing
        and expansion_plan.origin_to_family_rows is not None
        and expansion_plan.origin_to_base_row is not None
        and len(expansion_plan.origin_to_family_rows) >= b_origin
        and len(expansion_plan.origin_to_base_row) >= b_origin
    ):
        device = sampled_token_ids.device
        accepted_lens_t = torch.tensor(accepted_lens, dtype=torch.int32, device=device)

        row_to_prob = torch.full(
            (expansion_plan.expanded_batch_size,),
            float("-inf"),
            device=device,
            dtype=torch.float32,
        )
        row_to_rank = torch.full(
            (expansion_plan.expanded_batch_size,),
            10**9,
            device=device,
            dtype=torch.int32,
        )
        for fam in expansion_plan.families:
            for local_idx, row_idx in enumerate(fam.expanded_rows):
                j = int(row_idx)
                if 0 <= j < expansion_plan.expanded_batch_size:
                    row_to_prob[j] = float(fam.first_token_probs[local_idx])
                    row_to_rank[j] = int(fam.candidate_ranks[local_idx])

        family_rows = expansion_plan.origin_to_family_rows
        base_rows = expansion_plan.origin_to_base_row
        # origin_to_family_rows contains only extra rows; include base row explicitly.
        max_w = 1 + max((len(rows) for rows in family_rows[:b_origin]), default=0)
        idx_cpu = torch.zeros((b_origin, max_w), dtype=torch.long)
        valid_cpu = torch.zeros((b_origin, max_w), dtype=torch.bool)
        for o in range(b_origin):
            base = int(base_rows[o]) if o < len(base_rows) else -1
            col = 0
            if 0 <= base < expansion_plan.expanded_batch_size:
                idx_cpu[o, col] = base
                valid_cpu[o, col] = _pivot_packed_row_is_active(expansion_plan, base)
                col += 1

            rows = family_rows[o]
            for row_idx in rows:
                if col >= max_w:
                    break
                j = int(row_idx)
                if 0 <= j < expansion_plan.expanded_batch_size:
                    idx_cpu[o, col] = j
                    valid_cpu[o, col] = _pivot_packed_row_is_active(expansion_plan, j)
                else:
                    idx_cpu[o, col] = 0
                    valid_cpu[o, col] = False
                col += 1

        idx = idx_cpu.to(device=device)
        valid = valid_cpu.to(device=device)

        acc = accepted_lens_t.index_select(0, idx.view(-1)).view(b_origin, max_w)
        prob = row_to_prob.index_select(0, idx.view(-1)).view(b_origin, max_w)
        rank = row_to_rank.index_select(0, idx.view(-1)).view(b_origin, max_w)

        neg_inf_i = torch.full_like(acc, -1)
        neg_inf_f = torch.full_like(prob, float("-inf"))
        huge_i = torch.full_like(rank, 10**9)
        acc = torch.where(valid, acc, neg_inf_i)
        prob = torch.where(valid, prob, neg_inf_f)
        rank = torch.where(valid, rank, huge_i)

        best_acc = acc.max(dim=1, keepdim=True).values
        acc_mask = acc == best_acc
        prob2 = torch.where(acc_mask, prob, neg_inf_f)
        best_prob = prob2.max(dim=1, keepdim=True).values
        prob_mask = acc_mask & (prob == best_prob)
        neg_rank = torch.where(prob_mask, -rank, torch.full_like(rank, -(10**9)))
        best_col = neg_rank.argmax(dim=1)
        selected = idx[torch.arange(b_origin, device=device), best_col]
        return [int(x) for x in selected.tolist()]

    origin_to_family = {fam.origin_row: fam for fam in expansion_plan.families}
    selected_rows: list[int] = []
    for o in range(b_origin):
        fam = origin_to_family.get(o)
        if fam is None:
            if (
                expansion_plan.origin_to_base_row is not None
                and o < len(expansion_plan.origin_to_base_row)
                and int(expansion_plan.origin_to_base_row[o]) >= 0
            ):
                j = int(expansion_plan.origin_to_base_row[o])
            else:
                j = next(
                    idx
                    for idx, orig in enumerate(expansion_plan.expanded_to_origin)
                    if orig == o
                )
        else:
            best_row: int | None = None
            best_key = (-1, float("-inf"), 10**9)
            for local_idx, row_idx in enumerate(fam.expanded_rows):
                if not _pivot_packed_row_is_active(expansion_plan, row_idx):
                    continue
                if row_idx >= len(accepted_lens):
                    continue
                key = (
                    accepted_lens[row_idx],
                    fam.first_token_probs[local_idx],
                    -fam.candidate_ranks[local_idx],
                )
                if best_row is None or key > best_key:
                    best_key = key
                    best_row = row_idx
            if best_row is None:
                if (
                    expansion_plan.origin_to_base_row is not None
                    and o < len(expansion_plan.origin_to_base_row)
                    and int(expansion_plan.origin_to_base_row[o]) >= 0
                ):
                    j = int(expansion_plan.origin_to_base_row[o])
                else:
                    j = next(
                        idx
                        for idx, orig in enumerate(expansion_plan.expanded_to_origin)
                        if orig == o
                    )
            else:
                j = best_row
        selected_rows.append(j)
    return selected_rows


def collapse_pivot_expanded_sampled_to_origin(
    sampled_token_ids: torch.Tensor,
    expansion_plan: PivotExpansionPlan,
    *,
    num_draft_tokens: list[int] | None = None,
) -> torch.Tensor:
    """Map expanded verification rows (P) back to one row per origin request (B).

    Inactive fixed-capacity packed rows are never chosen from ``PivotExpansionFamily``
    (those indices are omitted from ``expanded_rows``); we still skip any inactive
    row defensively when ``packed_row_is_active`` is present.
    """
    if sampled_token_ids.shape[0] < expansion_plan.expanded_batch_size:
        return sampled_token_ids
    if not expansion_plan.expanded_to_origin:
        return sampled_token_ids
    selected_rows = select_pivot_expanded_rows_to_origin(
        sampled_token_ids,
        expansion_plan,
        num_draft_tokens=num_draft_tokens,
    )
    device = sampled_token_ids.device
    idx = torch.tensor(selected_rows, device=device, dtype=torch.long)
    return sampled_token_ids.index_select(0, idx)


def validate_root_only_pivot_expansion(
    *,
    prefix_rows: list[list[int]],
    expansion_plan: PivotExpansionPlan | None,
) -> DitDebugCheckResult:
    has_only_root_prefix = all(len(row) == 0 for row in prefix_rows)
    if expansion_plan is None:
        return DitDebugCheckResult(
            code="check_root_only_pivot_expansion",
            ok=True,
            detail="no expansion plan",
        )
    return DitDebugCheckResult(
        code="check_root_only_pivot_expansion",
        ok=has_only_root_prefix,
        detail=f"has_only_root_prefix={has_only_root_prefix}",
    )


def expand_intermediate_state_for_pivot_plan(
    state: IntermediateRoundState,
    plan: PivotExpansionPlan,
) -> IntermediateRoundState:
    """Expand provisional frontier metadata according to pivot expansion plan."""
    if state.frontier_metadata is None:
        return state
    if len(state.frontier_metadata) == plan.expanded_batch_size:
        return state
    sm = plan.packed_sm_origin
    if sm is None:
        sm = plan.expanded_to_origin
    expanded_frontier = [state.frontier_metadata[o] for o in sm]
    state.frontier_metadata = [dict(item) for item in expanded_frontier]
    return state


def get_unselected_cleanup_rows(
    *,
    expansion_plan: PivotExpansionPlan,
    selected_rows: list[int],
) -> list[int]:
    """Return expanded rows that must be cleaned after target collapse.

    Only **active** packed candidate rows are considered; inactive placeholder rows
    are never listed (they are not part of the family competition contract).
    """
    selected = set(int(r) for r in selected_rows)
    cleanup: list[int] = []
    for fam in expansion_plan.families:
        for row_idx in fam.expanded_rows:
            if not _pivot_packed_row_is_active(expansion_plan, row_idx):
                continue
            if row_idx not in selected:
                cleanup.append(int(row_idx))
    return cleanup


@dataclass(frozen=True)
class DitDebugCheckResult:
    code: str
    ok: bool
    detail: str


def validate_hierarchical_verification_tail_len(
    *,
    tail_len: int,
    interval_tokens: int,
    remaining_cap: int,
) -> DitDebugCheckResult:
    expected = min(interval_tokens, max(0, remaining_cap))
    return DitDebugCheckResult(
        code="check4_pre_target_tail_draft",
        ok=(tail_len == expected),
        detail=(
            f"tail_len={tail_len}, expected={expected}, "
            f"interval_tokens={interval_tokens}, remaining_cap={remaining_cap}"
        ),
    )


def validate_hierarchical_verification_source_stage_row(
    *,
    source_stage_row: list[int],
    bundle_len: int,
) -> DitDebugCheckResult:
    i_count = sum(1 for stage in source_stage_row if stage == 0)
    tail_count = sum(1 for stage in source_stage_row if stage == 1)
    has_only_known_stages = all(stage in (0, 1) for stage in source_stage_row)
    ok = has_only_known_stages and (i_count + tail_count == len(source_stage_row) == bundle_len)
    return DitDebugCheckResult(
        code="check8_source_stage_consistency",
        ok=ok,
        detail=(
            f"bundle_len={bundle_len}, source_len={len(source_stage_row)}, "
            f"i_count={i_count}, tail_count={tail_count}, "
            f"has_only_known_stages={has_only_known_stages}"
        ),
    )


def _compute_processed_sampling_logits(
    sampler: Sampler,
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    *,
    predict_bonus_token: bool = False,
) -> torch.Tensor:
    """Compute proposal logits after sampler processors and constraints."""
    logits = logits.to(torch.float32)
    logits = sampler.apply_logits_processors(logits, sampling_metadata, predict_bonus_token)
    if sampling_metadata.all_greedy:
        return logits

    assert sampling_metadata.temperature is not None
    temperature = sampling_metadata.temperature
    if not sampling_metadata.all_random:
        temperature = torch.where(temperature < _SAMPLING_EPS, 1.0, temperature)
    logits = logits.div(temperature.unsqueeze(dim=1))
    for processor in sampling_metadata.logitsprocs.argmax_invariant:
        logits = processor.apply(logits)
    return apply_top_k_top_p(logits, sampling_metadata.top_k, sampling_metadata.top_p)


def _sample_from_processed_logits(
    processed_logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    """Sample ids from processed logits using sampler-equivalent rules."""
    if sampling_metadata.all_random:
        greedy_sampled = None
    else:
        greedy_sampled = processed_logits.argmax(dim=-1).view(-1)
        if sampling_metadata.all_greedy:
            return greedy_sampled.to(torch.int32)

    probs = processed_logits.softmax(dim=-1, dtype=torch.float32)
    random_sampled = random_sample(probs, sampling_metadata.generators)
    if greedy_sampled is None:
        return random_sampled.to(torch.int32)
    assert sampling_metadata.temperature is not None
    sampled = torch.where(
        sampling_metadata.temperature < _SAMPLING_EPS,
        greedy_sampled,
        random_sampled,
    )
    return sampled.to(torch.int32)


def sample_next_token_and_probs_processed(
    sampler: Sampler,
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    *,
    predict_bonus_token: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample token ids and return processed proposal distribution q."""
    processed_logits = _compute_processed_sampling_logits(
        sampler,
        logits.to(torch.float32).clone(),
        sampling_metadata,
        predict_bonus_token=predict_bonus_token,
    )
    sampled = _sample_from_processed_logits(processed_logits.clone(), sampling_metadata)
    probs = processed_logits.softmax(dim=-1, dtype=torch.float32)
    return sampled, probs


def sample_greedy_with_constraints(
    sampler: Sampler,
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    *,
    predict_bonus_token: bool = False,
) -> torch.Tensor:
    """Compatibility helper returning shape [batch, 1] sampled token ids."""
    sampled, _ = sample_next_token_and_probs_processed(
        sampler,
        logits,
        sampling_metadata,
        predict_bonus_token=predict_bonus_token,
    )
    return sampled.unsqueeze(-1)


def run_verify_stage(
    rejection_sampler: RejectionSampler,
    *,
    draft_token_ids_flat: torch.Tensor,
    draft_probs_flat: torch.Tensor | None,
    num_draft_tokens: list[int],
    cu_num_draft_tokens: torch.Tensor,
    max_spec_len: int,
    verifier_logits_flat: torch.Tensor,
    bonus_logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    vocab_size: int,
    include_processed_probs: bool = False,
) -> tuple[list[list[int]], SamplerOutput, torch.Tensor, torch.Tensor]:
    """Run local verification via processed RejectionSampler.forward path."""
    batch_size = len(num_draft_tokens)
    num_tokens = int(draft_token_ids_flat.shape[0])
    device = draft_token_ids_flat.device

    logits = torch.cat([verifier_logits_flat, bonus_logits], dim=0)
    target_logits_indices = torch.arange(num_tokens, dtype=torch.int32, device=device)
    bonus_logits_indices = torch.arange(
        num_tokens, num_tokens + batch_size, dtype=torch.int32, device=device
    )
    logits_indices = torch.arange(
        num_tokens + batch_size, dtype=torch.int32, device=device
    )
    cu_num_sampled_tokens = torch.cumsum(
        torch.tensor(
            [n + 1 for n in num_draft_tokens], dtype=torch.int32, device=device
        ),
        dim=0,
    ).to(torch.int32)
    metadata = SpecDecodeMetadata(
        draft_token_ids=draft_token_ids_flat,
        num_draft_tokens=num_draft_tokens,
        cu_num_draft_tokens=cu_num_draft_tokens,
        cu_num_sampled_tokens=cu_num_sampled_tokens,
        target_logits_indices=target_logits_indices,
        bonus_logits_indices=bonus_logits_indices,
        logits_indices=logits_indices,
    )
    sm = replace(sampling_metadata, max_num_logprobs=None)
    if include_processed_probs:
        out, processed_target_probs, processed_bonus_probs = (
            rejection_sampler.forward_with_processed_probs(
                metadata,
                draft_probs_flat,
                logits,
                sm,
            )
        )
    else:
        out = rejection_sampler(metadata, draft_probs_flat, logits, sm)
        processed_target_probs = torch.empty(0, dtype=torch.float32, device=device)
        processed_bonus_probs = torch.empty(0, dtype=torch.float32, device=device)
    rows, _ = RejectionSampler.parse_output(out.sampled_token_ids, vocab_size)
    return rows, out, processed_target_probs, processed_bonus_probs


def _permute_pivot_expansion_plan(
    plan: PivotExpansionPlan,
    perm_old: list[int],
) -> PivotExpansionPlan:
    """Row ``j`` after permutation uses data from old row ``perm_old[j]``."""
    p = len(perm_old)
    inv = {perm_old[j]: j for j in range(p)}

    def pick_row_major(field: list[int] | None) -> list[int] | None:
        if field is None or len(field) != p:
            return field
        return [field[perm_old[j]] for j in range(p)]

    new_families: list[PivotExpansionFamily] = []
    for fam in plan.families:
        new_families.append(
            replace(
                fam,
                expanded_rows=[inv[r] for r in fam.expanded_rows],
            )
        )
    new_otf: list[list[int]] | None = None
    if plan.origin_to_family_rows is not None:
        new_otf = [[inv[r] for r in rows] for rows in plan.origin_to_family_rows]
    new_otb: list[int] | None = None
    if plan.origin_to_base_row is not None:
        new_otb = [inv[plan.origin_to_base_row[o]] for o in range(len(plan.origin_to_base_row))]

    if len(plan.expanded_to_origin) != p:
        return plan
    new_eto: list[int] = [plan.expanded_to_origin[perm_old[j]] for j in range(p)]
    return replace(
        plan,
        expanded_to_origin=new_eto,
        packed_sm_origin=pick_row_major(plan.packed_sm_origin),
        packed_to_origin=pick_row_major(plan.packed_to_origin),
        packed_row_is_active=pick_row_major(plan.packed_row_is_active),
        packed_row_is_base=pick_row_major(plan.packed_row_is_base),
        packed_row_family_rank=pick_row_major(plan.packed_row_family_rank),
        families=new_families,
        origin_to_family_rows=new_otf,
        origin_to_base_row=new_otb,
    )


def _reorder_hybrid_bundle_rows(
    bundle: HybridProposalBundle,
    perm_old: list[int],
) -> HybridProposalBundle:
    """Rebuild flattened tensors in row order ``perm_old`` (new row j <- old ``perm_old[j]``)."""
    p = len(perm_old)
    old_lens = bundle.num_draft_tokens
    m_old = len(old_lens)
    if p == 0:
        device = bundle.draft_token_ids.device
        return replace(
            bundle,
            draft_token_ids=bundle.draft_token_ids.new_empty((0,), dtype=torch.int32),
            draft_probs=(
                None
                if bundle.draft_probs is None
                else bundle.draft_probs.new_empty(
                    (0, bundle.draft_probs.shape[-1]), dtype=bundle.draft_probs.dtype
                )
            ),
            num_draft_tokens=[],
            cu_num_draft_tokens=torch.zeros(0, dtype=torch.int32, device=device),
            max_spec_len=bundle.max_spec_len,
            source_stage=(
                None
                if bundle.source_stage is None
                else bundle.source_stage.new_empty((0,), dtype=torch.int32)
            ),
            bundle_row_req_ids=(),
            expansion_plan=None,
            tree_plan=None,
        )
    for idx in perm_old:
        if idx < 0 or idx >= m_old:
            raise ValueError(
                f"invalid perm_old index {idx} for bundle with {m_old} rows: {perm_old}"
            )
    new_lens = [old_lens[perm_old[j]] for j in range(p)]
    tok_rows = _split_flat_tokens_by_lengths(bundle.draft_token_ids, old_lens)
    new_tok = torch.cat([tok_rows[perm_old[j]] for j in range(p)], dim=0).to(torch.int32)
    device = new_tok.device
    cu = torch.cumsum(torch.tensor(new_lens, dtype=torch.int32, device=device), dim=0).to(torch.int32)
    new_probs = None
    if bundle.draft_probs is not None and bundle.draft_probs.shape[0] == int(
        bundle.draft_token_ids.shape[0]
    ):
        prob_rows = _split_probs_by_lengths(bundle.draft_probs, old_lens)
        new_probs = torch.cat([prob_rows[perm_old[j]] for j in range(p)], dim=0)
    new_src = None
    if bundle.source_stage is not None and bundle.source_stage.shape[0] == int(
        bundle.draft_token_ids.shape[0]
    ):
        src_rows = _split_flat_tokens_by_lengths(bundle.source_stage, old_lens)
        new_src = torch.cat([src_rows[perm_old[j]] for j in range(p)], dim=0).to(
            torch.int32
        )
    br = bundle.bundle_row_req_ids
    new_br: tuple[str, ...] | None = None
    if br is not None and len(br) == m_old:
        new_br = tuple(str(br[perm_old[j]]) for j in range(p))
    new_plan = None
    if bundle.expansion_plan is not None:
        ep = bundle.expansion_plan
        if len(ep.expanded_to_origin) == p:
            new_plan = _permute_pivot_expansion_plan(ep, perm_old)
    max_spec_len = max(new_lens) if new_lens else bundle.max_spec_len
    return replace(
        bundle,
        draft_token_ids=new_tok,
        draft_probs=new_probs,
        num_draft_tokens=new_lens,
        cu_num_draft_tokens=cu,
        max_spec_len=max_spec_len,
        source_stage=new_src,
        bundle_row_req_ids=new_br,
        expansion_plan=new_plan,
        tree_plan=None if p != m_old else bundle.tree_plan,
    )


def remap_hybrid_bundle_rows_for_metadata(
    bundle: HybridProposalBundle,
    target_row_req_ids: list[str],
    *,
    spec_decode_metadata: SpecDecodeMetadata | None = None,
) -> tuple[HybridProposalBundle | None, str | None]:
    """Align bundle rows to ``target_row_req_ids`` (metadata row order).

    Supports:
    - **Same cardinality**: reorder rows so req-id multiset matches (permutation).
    - **Subset (batch shrink)**: metadata row count *n* can be smaller than the
      bundle *m* when every metadata req-id appears at least as often in the
      bundle (``Counter(metadata) ⊆ Counter(bundle)``). Surviving rows are
      picked in metadata order; leftover bundle rows (finished requests) are
      dropped.

    When *n < m* and ``spec_decode_metadata`` carries a pivot ``expansion_plan``
    with ``len == n``, it replaces the bundle plan after tensor rebuild so the
    plan matches the current prepare-time batch.

    Tree bundles: reorder is rejected; subset drops ``tree_plan``.
    """
    n = len(target_row_req_ids)
    m = len(bundle.num_draft_tokens)
    if n > m:
        return None, (
            f"metadata rows {n} exceed bundle rows {m} "
            "(cannot remap to a superset)"
        )
    br = bundle.bundle_row_req_ids
    if br is None:
        if n == m:
            return bundle, None
        return None, (
            "bundle_row_req_ids missing; cannot subset/remap "
            f"(bundle_rows={m}, metadata_rows={n})"
        )
    if len(br) != m:
        return None, "bundle_row_req_ids length mismatch"
    b_list = [str(x) for x in br]
    t_list = [str(x) for x in target_row_req_ids]
    if b_list == t_list:
        return bundle, None

    ct_b = Counter(b_list)
    ct_t = Counter(t_list)
    for r, need in ct_t.items():
        if ct_b[r] < need:
            return None, (
                f"metadata needs {need} row(s) for req_id={r!r}, "
                f"bundle has {ct_b[r]} "
                f"(bundle_counts={sorted(ct_b.items())}, "
                f"meta_counts={sorted(ct_t.items())})"
            )

    if bundle.expansion_plan is not None:
        ep = bundle.expansion_plan
        if len(ep.expanded_to_origin) != m:
            return None, (
                "expansion_plan expanded_to_origin length does not match bundle rows "
                f"({len(ep.expanded_to_origin)} vs {m})"
            )

    queues: dict[str, deque[int]] = {}
    for i, r in enumerate(b_list):
        queues.setdefault(r, deque()).append(i)
    perm_old: list[int] = []
    for r in t_list:
        q = queues.get(r)
        if not q:
            return None, f"exhausted bundle rows for req_id={r!r}"
        perm_old.append(q.popleft())

    if bundle.tree_plan is not None and n == m and perm_old != list(range(m)):
        return None, "cannot reorder hybrid bundle rows when tree_plan is set"

    subset = n < m
    if not subset and perm_old == list(range(m)):
        return bundle, None

    try:
        out = _reorder_hybrid_bundle_rows(bundle, perm_old)
    except ValueError as e:
        return None, str(e)
    if subset:
        out = replace(out, tree_plan=None)
        meta = spec_decode_metadata
        if meta is not None and len(meta.num_draft_tokens) == n:
            mep = meta.expansion_plan
            if mep is not None and len(mep.expanded_to_origin) == n:
                out = replace(out, expansion_plan=mep)
            else:
                out = replace(out, expansion_plan=None)
    return out, ("subset_survivor_remap" if subset else None)


def sanitize_hybrid_bundle_for_metadata(
    bundle: HybridProposalBundle,
    spec_decode_metadata: SpecDecodeMetadata,
    *,
    runner_num_spec_tokens: int,
) -> tuple[HybridProposalBundle | None, str | None]:
    """Validate/sanitize proposer bundle for current target verification metadata."""
    batch_size = len(spec_decode_metadata.num_draft_tokens)
    if len(bundle.num_draft_tokens) != batch_size:
        return None, (
            "num_draft_tokens batch mismatch "
            f"(bundle={len(bundle.num_draft_tokens)}, metadata={batch_size})"
        )
    if bundle.cu_num_draft_tokens.numel() != batch_size:
        return None, (
            "cu_num_draft_tokens length mismatch "
            f"(bundle={bundle.cu_num_draft_tokens.numel()}, metadata={batch_size})"
        )
    expected_total = int(spec_decode_metadata.cu_num_draft_tokens[-1].item())
    bundle_total = int(bundle.cu_num_draft_tokens[-1].item()) if batch_size > 0 else 0
    if bundle_total != expected_total:
        return None, (
            "flattened token count mismatch "
            f"(bundle={bundle_total}, metadata={expected_total})"
        )
    if bundle.draft_token_ids.shape[0] != expected_total:
        return None, (
            "draft_token_ids rows mismatch "
            f"(bundle={bundle.draft_token_ids.shape[0]}, metadata={expected_total})"
        )
    if not torch.equal(bundle.draft_token_ids, spec_decode_metadata.draft_token_ids):
        return None, "draft_token_ids content does not match metadata"

    sanitized = bundle
    if sanitized.draft_probs is not None:
        probs = sanitized.draft_probs
        if probs.ndim != 2 or probs.shape[0] != expected_total:
            sanitized = replace(sanitized, draft_probs=None)
    if sanitized.source_stage is not None:
        source_stage = sanitized.source_stage
        if source_stage.ndim != 1 or source_stage.shape[0] != expected_total:
            sanitized = replace(sanitized, source_stage=None)
    if sanitized.expansion_plan is not None:
        plan = sanitized.expansion_plan
        if plan.expanded_batch_size < 0 or len(plan.expanded_to_origin) != plan.expanded_batch_size:
            sanitized = replace(sanitized, expansion_plan=None)
        elif plan.packed_sm_origin is not None and len(plan.packed_sm_origin) != plan.expanded_batch_size:
            sanitized = replace(sanitized, expansion_plan=None)
        elif (
            plan.packed_row_is_active is not None
            and len(plan.packed_row_is_active) != plan.expanded_batch_size
        ):
            sanitized = replace(sanitized, expansion_plan=None)
        elif plan.uses_fixed_capacity_packing and (
            plan.packed_sm_origin is None
            or len(plan.packed_sm_origin) != plan.expanded_batch_size
        ):
            sanitized = replace(sanitized, expansion_plan=None)
        elif not plan.uses_fixed_capacity_packing and len(plan.families) == 0:
            sanitized = replace(sanitized, expansion_plan=None)
    if sanitized.max_spec_len != runner_num_spec_tokens:
        # Keep bundle usable; width mismatch is informational only.
        return sanitized, (
            f"max_spec_len mismatch (bundle={sanitized.max_spec_len}, "
            f"runner={runner_num_spec_tokens})"
        )
    return sanitized, None


