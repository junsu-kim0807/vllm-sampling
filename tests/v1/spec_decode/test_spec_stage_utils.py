# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace

import torch

from vllm.v1.sample.logits_processor.state import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.spec_stage_utils import (
    materialize_spec_token_ids_from_prefix_tensors,
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


def test_slice_provisional_prefix_tensors_no_penalties_keeps_gpu_prefix() -> None:
    """HV-style path: no list materialization when penalties/bad-words are off."""
    sm = _minimal_sm(batch_size=2, output_token_ids=[[1], [2]])
    pt = torch.tensor([[3, 4, 0], [5, 0, 0]], dtype=torch.int32)
    pl = torch.tensor([2, 1], dtype=torch.int32)
    out = slice_sampling_metadata_for_subbatch(
        sm,
        [1, 0],
        provisional_prefix_tokens=pt,
        provisional_prefix_lens=pl,
        sampled_ids_only=True,
    )
    assert out.spec_token_ids is None
    assert out.spec_prefix_tokens is not None and out.spec_prefix_lens is not None
    assert out.spec_prefix_tokens.shape == (2, 3)
    assert list(out.spec_prefix_lens.tolist()) == [1, 2]


def test_slice_provisional_prefix_tensors_with_bad_words_keeps_gpu_prefix() -> None:
    """Bad-words / penalties: prefix stays on device; list form is materialized at RS."""
    sm = _minimal_sm(batch_size=2, output_token_ids=[[1], [2]])
    sm = replace(sm, bad_words_token_ids={0: [[999]]})
    pt = torch.tensor([[3, 4], [5, 6]], dtype=torch.int32)
    pl = torch.tensor([2, 1], dtype=torch.int32)
    out = slice_sampling_metadata_for_subbatch(
        sm,
        [0],
        provisional_prefix_tokens=pt,
        provisional_prefix_lens=pl,
    )
    assert out.spec_token_ids is None
    assert out.spec_prefix_tokens is not None and out.spec_prefix_lens is not None
    assert out.spec_prefix_tokens.shape == (1, 2)
    assert list(out.spec_prefix_lens.tolist()) == [2]
    assert materialize_spec_token_ids_from_prefix_tensors(
        out.spec_prefix_tokens, out.spec_prefix_lens
    ) == [[3, 4]]
