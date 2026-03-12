# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.starkv.superpress import SuperPress


def test_superpress_score_prefill_returns_dict():
    press = SuperPress()
    out = press.score_prefill({"request_ids": ["a"], "num_tokens": 50})
    assert "confidence_by_request" in out
    assert "low_confidence_requests" in out
    assert "a" in out["confidence_by_request"]


def test_superpress_score_decode_respects_max_reforward_steps():
    press = SuperPress()
    press.confidence_threshold = 0.9
    press.max_reforward_steps = 1
    payload = {
        "confidence_by_request": {"r1": 0.5},
        "current_pos_by_request": {"r1": 10},
        "last_reforward_index_by_request": {},
        "prompt_len_by_request": {"r1": 2},
    }
    first = press.score_decode(payload)
    assert len(first["plans"]) == 1
    second = press.score_decode(payload)
    assert len(second["plans"]) == 0
