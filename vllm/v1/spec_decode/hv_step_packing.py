# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packing helpers for hierarchical verification.

Standalone HV **verify** uses intermediate frontier metadata + direct forward and
does not call these helpers on the hot path. Standalone HV **draft** uses
``HierarchicalVerificationProposer.propose_chunk_from_prefix`` →
``GPUModelRunner._run_hv_draft_step_from_frontier`` (draft frontier + metadata-direct
``_prepare_inputs``) when draft KV frontier mode is enabled, so it does not call
``build_prefix_conditioned_inputs`` on that hot path. Adaptive/pivot and other
legacy paths still use ``build_prefix_conditioned_inputs`` via
``GPUModelRunner._run_hv_draft_step`` (runner seam, not
``adaptive_cascade._build_prefix_conditioned_inputs``). Adaptive/pivot
prefix-conditioned verify and unit tests continue to use this module.
"""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

logger = init_logger(__name__)


def _hv_debug_enabled() -> bool:
    return (
        os.environ.get("VLLM_SPEC_SPECHIVE_DEBUG", "0") == "1"
        or os.environ.get("VLLM_SPEC_DIT_DEBUG", "0") == "1"
    )


def _hv_debug_assert(cond: bool, code: str, *, detail: str = "") -> bool:
    if _hv_debug_enabled() and not cond:
        logger.warning("HV_DEBUG check failed: %s (%s)", code, detail)
    return cond


def build_prefix_conditioned_inputs(
    *,
    cad: CommonAttentionMetadata,
    target_token_ids: torch.Tensor,
    target_positions: torch.Tensor,
    target_hidden_states: torch.Tensor,
    next_token_ids: torch.Tensor,
    prefix_rows: list[list[int]],
    roll_rows: list[list[int]] | None,
    block_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    CommonAttentionMetadata,
    list[int],
    list[int],
    list[int],
]:
    """Build proposer inputs conditioned on logical prefix rows per request.

    Same geometry as ``adaptive_cascade._build_prefix_conditioned_inputs`` but
    parameterized by ``block_size`` so the runner can call it without a
    full ``SpecDecodeBaseProposer`` wrapper.
    """
    batch_size = cad.batch_size()
    assert len(prefix_rows) == batch_size
    if roll_rows is not None:
        assert len(roll_rows) == batch_size
    qsl = cad.query_start_loc
    orig_seq_lens = cad.seq_lens.clone()
    device = target_token_ids.device
    hidden_size = target_hidden_states.shape[-1]

    token_pieces: list[torch.Tensor] = []
    pos_pieces_1d: list[torch.Tensor] = []
    pos_pieces_2d: list[torch.Tensor] = []
    hidden_pieces: list[torch.Tensor] = []
    slot_pieces: list[torch.Tensor] = []
    next_out: list[int] = []
    seq_lens = cad.seq_lens.clone()
    new_qsl = [0]
    max_query_len = 1
    base_query_lens: list[int] = []
    prefix_lens: list[int] = []
    roll_lens: list[int] = []

    for b in range(batch_size):
        s = int(qsl[b].item())
        e = int(qsl[b + 1].item())
        base_query_lens.append(e - s)
        base_next = int(next_token_ids[b].item())
        logical = prefix_rows[b]
        prefix_lens.append(len(logical))
        forced = roll_rows[b] if roll_rows is not None else []
        roll_lens.append(len(forced))
        chain = [base_next, *logical, *forced]
        ext = chain[:-1]
        next_tok = chain[-1]
        if _hv_debug_enabled() and b < 2:
            logger.info(
                "HV_DEBUG packing req=%d base_next=%s logical=%s forced=%s ext=%s next_tok=%s",
                b,
                base_next,
                logical,
                forced,
                ext,
                next_tok,
            )

        tok_piece = target_token_ids[s:e]
        hid_piece = target_hidden_states[s:e]
        slot_piece = cad.slot_mapping[s:e]
        ext_len = len(ext)
        if ext_len > 0:
            ext_tensor = torch.tensor(ext, dtype=torch.int32, device=device)
            tok_piece = torch.cat((tok_piece, ext_tensor), dim=0)
            hid_piece = torch.cat(
                (
                    hid_piece,
                    torch.zeros(
                        (ext_len, hidden_size),
                        dtype=target_hidden_states.dtype,
                        device=target_hidden_states.device,
                    ),
                ),
                dim=0,
            )
            seq_base = int(cad.seq_lens[b].item())
            ext_positions = torch.arange(
                seq_base,
                seq_base + ext_len,
                dtype=torch.int64,
                device=device,
            )
            block_numbers = torch.div(
                ext_positions, block_size, rounding_mode="floor"
            ).to(torch.long)
            block_ids = cad.block_table_tensor[b].gather(0, block_numbers)
            ext_slot = block_ids * block_size + torch.remainder(
                ext_positions, block_size
            ).to(block_ids.dtype)
            slot_piece = torch.cat(
                (slot_piece, ext_slot.to(slot_piece.dtype)),
                dim=0,
            )
            _hv_debug_assert(
                bool((ext_slot >= 0).all().item()),
                "check_hv_packing_slot_mapping_non_negative",
                detail=f"req={b}, ext_len={ext_len}",
            )

        if target_positions.dim() == 1:
            pos_piece = target_positions[s:e]
            if ext_len > 0:
                ext_pos = torch.arange(
                    int(pos_piece[-1].item()) + 1,
                    int(pos_piece[-1].item()) + 1 + ext_len,
                    dtype=target_positions.dtype,
                    device=target_positions.device,
                )
                pos_piece = torch.cat((pos_piece, ext_pos), dim=0)
            pos_pieces_1d.append(pos_piece)
        else:
            pos_piece_2d = target_positions[:, s:e]
            if ext_len > 0:
                increments = torch.arange(
                    1,
                    ext_len + 1,
                    dtype=target_positions.dtype,
                    device=target_positions.device,
                ).view(1, -1)
                ext_pos_2d = pos_piece_2d[:, -1:].repeat(1, ext_len) + increments
                pos_piece_2d = torch.cat((pos_piece_2d, ext_pos_2d), dim=1)
            pos_pieces_2d.append(pos_piece_2d)

        token_pieces.append(tok_piece)
        hidden_pieces.append(hid_piece)
        slot_pieces.append(slot_piece)
        next_out.append(next_tok)
        seq_lens[b] = seq_lens[b] + ext_len
        _hv_debug_assert(
            int(seq_lens[b].item()) == int(orig_seq_lens[b].item()) + ext_len,
            "check_hv_packing_seq_lens_update",
            detail=(
                f"req={b}, orig={int(orig_seq_lens[b].item())}, "
                f"ext_len={ext_len}, updated={int(seq_lens[b].item())}"
            ),
        )
        query_len = (e - s) + ext_len
        max_query_len = max(max_query_len, query_len)
        new_qsl.append(new_qsl[-1] + query_len)

    out_tokens = torch.cat(token_pieces, dim=0)
    out_hidden = torch.cat(hidden_pieces, dim=0)
    out_slot = torch.cat(slot_pieces, dim=0)
    if target_positions.dim() == 1:
        out_positions = torch.cat(pos_pieces_1d, dim=0)
    else:
        out_positions = torch.cat(pos_pieces_2d, dim=1)
    out_next = torch.tensor(next_out, dtype=torch.int32, device=next_token_ids.device)
    out_qsl = torch.tensor(new_qsl, dtype=qsl.dtype, device=qsl.device)
    out_qsl_cpu = out_qsl.detach().cpu()
    out_seq_lens_cpu = seq_lens.detach().cpu() if cad._seq_lens_cpu is not None else None
    out_cad = cad.replace(
        query_start_loc=out_qsl,
        query_start_loc_cpu=out_qsl_cpu,
        seq_lens=seq_lens,
        num_reqs=batch_size,
        num_actual_tokens=int(out_tokens.shape[0]),
        max_query_len=max_query_len,
        max_seq_len=int(seq_lens.max().item()),
        slot_mapping=out_slot,
        _seq_lens_cpu=out_seq_lens_cpu,
        _num_computed_tokens_cpu=None,
        _num_computed_tokens_cache=None,
    )
    _hv_debug_assert(
        int(new_qsl[-1]) == int(out_tokens.shape[0]),
        "check_hv_packing_new_qsl_total_tokens",
        detail=f"new_qsl_end={new_qsl[-1]}, total_tokens={out_tokens.shape[0]}",
    )
    return (
        out_tokens,
        out_positions,
        out_hidden,
        out_next,
        out_cad,
        base_query_lens,
        prefix_lens,
        roll_lens,
    )


def slice_hv_verification_logits(
    all_logits: torch.Tensor,
    pref_cad: CommonAttentionMetadata,
    draft_tokens: torch.Tensor,
    base_query_lens: list[int],
    prefix_lens: list[int],
    roll_lens: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice verifier logits for chunk + bonus rows (same layout as legacy HV)."""
    bsz = draft_tokens.shape[0]
    chunk_len = draft_tokens.shape[1]
    qsl = pref_cad.query_start_loc
    per_req_steps: list[torch.Tensor] = []
    bonus_rows: list[torch.Tensor] = []
    for b in range(bsz):
        chain_len = prefix_lens[b] + roll_lens[b] + 1
        start = int(qsl[b].item()) + base_query_lens[b] - 1
        idx = torch.arange(
            start,
            start + chain_len,
            dtype=torch.long,
            device=all_logits.device,
        )
        chain_logits = all_logits[idx]
        verify_start = prefix_lens[b]
        verify_end = verify_start + chunk_len
        bonus_index = verify_end
        _hv_debug_assert(
            verify_start >= 0
            and verify_end <= chain_logits.shape[0] - 1
            and bonus_index < chain_logits.shape[0],
            "check_hv_verify_slice_bounds",
            detail=(
                f"req={b}, base_query_len={base_query_lens[b]}, "
                f"prefix_len={prefix_lens[b]}, chunk_len={chunk_len}, "
                f"verify_start={verify_start}, verify_end={verify_end}, "
                f"bonus_index={bonus_index}, chain_rows={chain_logits.shape[0]}"
            ),
        )
        per_req_steps.append(chain_logits[verify_start:verify_end])
        bonus_rows.append(chain_logits[bonus_index])

    per_req = torch.stack(per_req_steps, dim=0).to(torch.float32)
    bonus_logits = torch.stack(bonus_rows, dim=0).to(torch.float32)
    return per_req.reshape(-1, per_req.shape[-1]), bonus_logits


def gather_hv_verification_logits_from_spec_decode_metadata(
    all_logits: torch.Tensor,
    spec_decode_metadata: SpecDecodeMetadata,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick (logits_flat, bonus_logits) for HV verify from one model forward.

    ``all_logits`` is indexed the same way as standard spec-decode verification:
    rows selected by ``target_logits_indices`` / ``bonus_logits_indices`` must
    align with ``_calc_spec_decode_metadata`` when the intermediate frontier
    batch matches the scheduler step geometry.
    """
    tid = spec_decode_metadata.target_logits_indices.long()
    bid = spec_decode_metadata.bonus_logits_indices.long()
    return all_logits[tid].to(torch.float32), all_logits[bid].to(torch.float32)
