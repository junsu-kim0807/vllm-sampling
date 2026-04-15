# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.sample.logits_processor.state import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.spec_stage_utils import (
    materialize_prefix_rows_from_dense,
    slice_sampling_metadata_for_subbatch,
)


def _minimal_sm(
    *,
    batch_size: int,
    output_token_ids: list[list[int]],
    spec_token_ids: list[list[int]] | None = None,
    device: torch.device | None = None,
) -> SamplingMetadata:
    dev = device or torch.device("cpu")
    z = torch.zeros(batch_size, device=dev)
    o = torch.ones(batch_size, device=dev)
    return SamplingMetadata(
        temperature=None,
        all_greedy=True,
        all_random=False,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=z.clone(),
        presence_penalties=z.clone(),
        repetition_penalties=o.clone(),
        output_token_ids=output_token_ids,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
        spec_token_ids=spec_token_ids,
    )


def test_slice_subbatch_empty_output_token_ids_synthesizes_rows() -> None:
    sm = _minimal_sm(batch_size=2, output_token_ids=[])
    out = slice_sampling_metadata_for_subbatch(sm, [0, 0, 1])
    assert out.output_token_ids == [[], [], []]


def test_slice_subbatch_nonempty_output_token_ids_unchanged_semantics() -> None:
    sm = _minimal_sm(
        batch_size=2,
        output_token_ids=[[1], [2, 3]],
    )
    out = slice_sampling_metadata_for_subbatch(sm, [1, 0])
    assert out.output_token_ids == [[2, 3], [1]]


def test_slice_subbatch_empty_spec_token_ids_synthesizes_rows() -> None:
    sm = _minimal_sm(batch_size=2, output_token_ids=[], spec_token_ids=[])
    out = slice_sampling_metadata_for_subbatch(sm, [0, 0])
    assert out.output_token_ids == [[], []]
    assert out.spec_token_ids == [[], []]


def test_materialize_prefix_rows_from_dense_roundtrip() -> None:
    tok = torch.tensor([[1, 2, 0], [3, 0, 0]], dtype=torch.int32)
    ln = torch.tensor([2, 1], dtype=torch.long)
    rows = materialize_prefix_rows_from_dense(tok, ln, 2)
    assert rows == [[1, 2], [3]]


def test_slice_subbatch_dense_provisional_matches_list() -> None:
    sm = _minimal_sm(batch_size=2, output_token_ids=[])
    tok = torch.tensor([[5, 0], [6, 7]], dtype=torch.int32)
    ln = torch.tensor([1, 2], dtype=torch.long)
    out_dense = slice_sampling_metadata_for_subbatch(
        sm,
        [0, 1],
        provisional_prefix_tokens=tok,
        provisional_prefix_lengths=ln,
        sampled_ids_only=True,
    )
    out_list = slice_sampling_metadata_for_subbatch(
        sm,
        [0, 1],
        provisional_prefix_rows=[[5], [6, 7]],
        sampled_ids_only=True,
    )
    assert out_dense.spec_token_ids == out_list.spec_token_ids


def test_slice_subbatch_sampled_ids_only_fast_path_smoke() -> None:
    """Greedy-style empty histories + ``sampled_ids_only`` use the fast slice path."""
    sm = _minimal_sm(batch_size=3, output_token_ids=[])
    idxs = [2, 0, 1]
    out = slice_sampling_metadata_for_subbatch(
        sm, idxs, sampled_ids_only=True
    )
    assert out.max_num_logprobs is None
    assert out.output_token_ids == [[], [], []]
    idx = torch.tensor(idxs, dtype=torch.long, device=sm.frequency_penalties.device)
    assert torch.equal(out.frequency_penalties, sm.frequency_penalties.index_select(0, idx))
