# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.sample.logits_processor.state import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.spec_stage_utils import slice_sampling_metadata_for_subbatch


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
