# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generic stage operations reused by adaptive/staged speculative decoding."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p, random_sample
from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID, RejectionSampler
from vllm.v1.sample.sampler import Sampler, _SAMPLING_EPS
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.spec_stage_runtime import (
    HybridProposalBundle,
    PivotExpansionPlan,
)


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


def collapse_pivot_expanded_sampled_to_origin(
    sampled_token_ids: torch.Tensor,
    expansion_plan: PivotExpansionPlan,
    *,
    num_draft_tokens: list[int] | None = None,
) -> torch.Tensor:
    """Map expanded verification rows (B') back to one row per origin request (B)."""
    if sampled_token_ids.shape[0] < expansion_plan.expanded_batch_size:
        return sampled_token_ids
    if not expansion_plan.expanded_to_origin:
        return sampled_token_ids
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
    b_origin = max(expansion_plan.expanded_to_origin) + 1
    origin_to_family = {fam.origin_row: fam for fam in expansion_plan.families}
    selected_rows: list[int] = []
    for o in range(b_origin):
        fam = origin_to_family.get(o)
        if fam is None:
            j = next(
                idx
                for idx, orig in enumerate(expansion_plan.expanded_to_origin)
                if orig == o
            )
        else:
            best_row = fam.expanded_rows[0]
            best_key = (-1, float("-inf"), 10**9)
            for local_idx, row_idx in enumerate(fam.expanded_rows):
                if row_idx >= len(accepted_lens):
                    continue
                key = (
                    accepted_lens[row_idx],
                    fam.first_token_probs[local_idx],
                    -fam.candidate_ranks[local_idx],
                )
                if key > best_key:
                    best_key = key
                    best_row = row_idx
            j = best_row
        selected_rows.append(j)
    device = sampled_token_ids.device
    idx = torch.tensor(selected_rows, device=device, dtype=torch.long)
    return sampled_token_ids.index_select(0, idx)


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
    )
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
        if (
            plan.expanded_batch_size < 0
            or len(plan.expanded_to_origin) != plan.expanded_batch_size
            or len(plan.families) == 0
        ):
            sanitized = replace(sanitized, expansion_plan=None)
    if sanitized.max_spec_len != runner_num_spec_tokens:
        # Keep bundle usable; width mismatch is informational only.
        return sanitized, (
            f"max_spec_len mismatch (bundle={sanitized.max_spec_len}, "
            f"runner={runner_num_spec_tokens})"
        )
    return sanitized, None


