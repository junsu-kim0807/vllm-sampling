# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for staged speculative decoding controllers."""

from __future__ import annotations

from dataclasses import replace

import torch

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.sample.metadata import SamplingMetadata


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
    sampled_ids_only: bool = False,
) -> SamplingMetadata:
    """Slice request-major SamplingMetadata for sub-batch stage execution."""
    idx = torch.tensor(idxs, device=sm.frequency_penalties.device, dtype=torch.long)

    def _slice_opt_tensor(x: torch.Tensor | None) -> torch.Tensor | None:
        if x is None:
            return None
        return x.index_select(0, idx)

    output_token_ids = [sm.output_token_ids[i] for i in idxs]
    if provisional_prefix_rows is not None:
        spec_token_ids = provisional_prefix_rows
    elif sm.spec_token_ids is not None:
        spec_token_ids = [sm.spec_token_ids[i] for i in idxs]
    else:
        spec_token_ids = None

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
        max_num_logprobs=None if sampled_ids_only else sm.max_num_logprobs,
    )
