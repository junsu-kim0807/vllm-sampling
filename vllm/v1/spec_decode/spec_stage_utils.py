# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for staged speculative decoding controllers."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.sample.metadata import SamplingMetadata


def materialize_spec_token_ids_from_prefix_tensors(
    spec_prefix_tokens: torch.Tensor,
    spec_prefix_lens: torch.Tensor,
) -> list[list[int]]:
    """CPU list rows from ``[B, cap]`` + ``[B]`` prefix tensors (for penalty / bad-words)."""
    pt = spec_prefix_tokens.detach().cpu().numpy()
    pl = spec_prefix_lens.detach().cpu().numpy().astype(np.int64, copy=False)
    out: list[list[int]] = []
    for i in range(pt.shape[0]):
        n = int(pl[i])
        if n <= 0:
            out.append([])
        else:
            out.append([int(x) for x in pt[i, :n].tolist()])
    return out


def group_indices_by_prefix_len(rows: list[list[int]]) -> list[tuple[int, list[int]]]:
    groups: dict[int, list[int]] = {}
    for b, row in enumerate(rows):
        p = len(row)
        groups.setdefault(p, []).append(b)
    return sorted(groups.items(), key=lambda x: x[0])


def clone_common_attn_metadata(cad: CommonAttentionMetadata) -> CommonAttentionMetadata:
    """Deep-clone tensor fields so local stage execution cannot mutate caller state."""

    def _c(t: torch.Tensor | None) -> torch.Tensor | None:
        return None if t is None else t.clone()

    return cad.replace(
        query_start_loc=_c(cad.query_start_loc),
        query_start_loc_cpu=_c(cad.query_start_loc_cpu),
        seq_lens=_c(cad.seq_lens),
        block_table_tensor=_c(cad.block_table_tensor),
        slot_mapping=_c(cad.slot_mapping),
        logits_indices_padded=_c(cad.logits_indices_padded),
        encoder_seq_lens=_c(cad.encoder_seq_lens),
        dcp_local_seq_lens=_c(cad.dcp_local_seq_lens),
        _seq_lens_cpu=None,
        _num_computed_tokens_cpu=None,
        _num_computed_tokens_cache=None,
    )


def slice_sampling_metadata_for_subbatch(
    sm: SamplingMetadata,
    idxs: list[int],
    *,
    provisional_prefix_rows: list[list[int]] | None = None,
    provisional_prefix_tokens: torch.Tensor | None = None,
    provisional_prefix_lens: torch.Tensor | None = None,
    sampled_ids_only: bool = False,
) -> SamplingMetadata:
    """Slice request-major SamplingMetadata for sub-batch stage execution."""
    idx = torch.tensor(idxs, device=sm.frequency_penalties.device, dtype=torch.long)

    def _slice_opt_tensor(x: torch.Tensor | None) -> torch.Tensor | None:
        if x is None:
            return None
        return x.index_select(0, idx)

    # Greedy + no-penalty batches omit output_token_ids in InputBatch; staged
    # pivot/spec paths still slice per logical row — synthesize empty histories.
    if sm.output_token_ids:
        output_token_ids = [sm.output_token_ids[i] for i in idxs]
    else:
        output_token_ids = [[] for _ in idxs]

    bad_words = {
        new_i: sm.bad_words_token_ids[old_i]
        for new_i, old_i in enumerate(idxs)
        if old_i in sm.bad_words_token_ids
    }
    generators = {
        new_i: sm.generators[old_i]
        for new_i, old_i in enumerate(idxs)
        if old_i in sm.generators
    }
    spec_token_ids: list[list[int]] | None = None
    spec_prefix_tokens: torch.Tensor | None = None
    spec_prefix_lens: torch.Tensor | None = None

    if provisional_prefix_tokens is not None and provisional_prefix_lens is not None:
        idx_dev = idx.to(
            device=provisional_prefix_tokens.device, dtype=idx.dtype
        )
        # Keep prefix on device; RS / sampler materialize once when penalties/bad-words
        # need list form (avoids duplicate GPU→CPU in this helper).
        spec_prefix_tokens = provisional_prefix_tokens.index_select(0, idx_dev)
        spec_prefix_lens = provisional_prefix_lens.index_select(0, idx_dev)
        spec_token_ids = None
    elif provisional_prefix_rows is not None:
        spec_prefix_tokens = None
        spec_prefix_lens = None
        spec_token_ids = provisional_prefix_rows
    elif sm.spec_token_ids is not None:
        spec_prefix_tokens = None
        spec_prefix_lens = None
        if sm.spec_token_ids:
            spec_token_ids = [sm.spec_token_ids[i] for i in idxs]
        else:
            spec_token_ids = [[] for _ in idxs]
    else:
        spec_token_ids = None
        if sm.spec_prefix_tokens is not None and sm.spec_prefix_lens is not None:
            idx_dev = idx.to(device=sm.spec_prefix_tokens.device, dtype=idx.dtype)
            spec_prefix_tokens = sm.spec_prefix_tokens.index_select(0, idx_dev)
            spec_prefix_lens = sm.spec_prefix_lens.index_select(0, idx_dev)
        else:
            spec_prefix_tokens = None
            spec_prefix_lens = None

    return replace(
        sm,
        temperature=_slice_opt_tensor(sm.temperature),
        top_p=_slice_opt_tensor(sm.top_p),
        top_k=_slice_opt_tensor(sm.top_k),
        frequency_penalties=sm.frequency_penalties.index_select(0, idx),
        presence_penalties=sm.presence_penalties.index_select(0, idx),
        repetition_penalties=sm.repetition_penalties.index_select(0, idx),
        prompt_token_ids=_slice_opt_tensor(sm.prompt_token_ids),
        allowed_token_ids_mask=_slice_opt_tensor(sm.allowed_token_ids_mask),
        output_token_ids=output_token_ids,
        bad_words_token_ids=bad_words,
        generators=generators,
        spec_token_ids=spec_token_ids,
        spec_prefix_tokens=spec_prefix_tokens,
        spec_prefix_lens=spec_prefix_lens,
        max_num_logprobs=None if sampled_ids_only else sm.max_num_logprobs,
    )
