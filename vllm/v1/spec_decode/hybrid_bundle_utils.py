# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared helpers to build ``HybridProposalBundle`` from padded draft rows."""

from __future__ import annotations

import torch

from vllm.logger import init_logger
from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID
from vllm.v1.spec_decode.spec_stage_runtime import HybridProposalBundle

logger = init_logger(__name__)


def _build_hybrid_bundle_from_rows(
    rows_2d: torch.Tensor,
    *,
    mode: str,
    draft_probs: torch.Tensor | None,
    source_stage_2d: torch.Tensor | None = None,
    bundle_row_req_ids: tuple[str, ...] | list[str] | None = None,
) -> HybridProposalBundle:
    """Build a flattened proposal bundle from fixed-width padded rows.

    Uses mask-based vectorised ops instead of a per-row Python loop to
    avoid repeated tensor slicing, list appends, and torch.cat overhead.

    Args:
        source_stage_2d: Optional int32 tensor with the same shape as
            *rows_2d*.  Valid positions (where rows_2d != PLACEHOLDER)
            carry the stage tag; padding positions are ignored.
    """
    num_rows = int(rows_2d.shape[0])

    br_ids: tuple[str, ...] | None = None
    if bundle_row_req_ids is not None:
        br_list = list(bundle_row_req_ids)
        if len(br_list) != num_rows:
            logger.warning(
                "bundle_row_req_ids length %s != num_rows %s; omitting req ids on bundle.",
                len(br_list),
                num_rows,
            )
        else:
            br_ids = tuple(str(x) for x in br_list)

    # --- vectorised valid-token extraction ---
    valid_mask = rows_2d.ne(PLACEHOLDER_TOKEN_ID)
    lengths_t = valid_mask.sum(dim=1, dtype=torch.int32)
    lengths = lengths_t.tolist()

    draft_token_ids = rows_2d.masked_select(valid_mask).to(torch.int32)
    cu = torch.cumsum(lengths_t, dim=0, dtype=torch.int32)

    # --- source_stage: single masked_select, no Python loop ---
    source_stage: torch.Tensor | None = None
    if source_stage_2d is not None:
        if source_stage_2d.shape != rows_2d.shape:
            raise ValueError(
                "source_stage_2d shape must match rows_2d shape: "
                f"{tuple(source_stage_2d.shape)} != {tuple(rows_2d.shape)}"
            )
        source_stage = source_stage_2d.masked_select(valid_mask).to(torch.int32)

    if draft_probs is not None and int(draft_probs.shape[0]) != int(
        draft_token_ids.shape[0]
    ):
        raise ValueError(
            "draft_probs first dimension must equal flattened draft token count: "
            f"{int(draft_probs.shape[0])} != {int(draft_token_ids.shape[0])}"
        )

    return HybridProposalBundle(
        draft_token_ids=draft_token_ids,
        draft_probs=draft_probs,
        num_draft_tokens=lengths,
        cu_num_draft_tokens=cu,
        max_spec_len=int(rows_2d.shape[1]),
        mode=mode,  # type: ignore[arg-type]
        source_stage=source_stage,
        bundle_row_req_ids=br_ids,
    )


def _flatten_prob_rows_for_output(
    prob_rows: list[list[torch.Tensor]],
    out_rows: torch.Tensor,
) -> torch.Tensor | None:
    """Flatten per-request per-token probs aligned to valid output rows."""
    flat: list[torch.Tensor] = []
    for b in range(out_rows.shape[0]):
        valid = int((out_rows[b] != PLACEHOLDER_TOKEN_ID).sum().item())
        if valid <= 0:
            continue
        if len(prob_rows[b]) < valid:
            return None
        flat.extend(prob_rows[b][:valid])
    if not flat:
        return None
    return torch.stack(flat, dim=0).to(torch.float32)
