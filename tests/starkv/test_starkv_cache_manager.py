# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.core.starkv_cache_manager import StarKVCacheManager, StarKVStats
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import ModelRunnerOutput


def _minimal_kv_cache_config():
    spec = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=64, dtype=torch.float32)
    group = KVCacheGroupSpec(layer_names=["layer"], kv_cache_spec=spec)
    return KVCacheConfig(num_blocks=64, kv_cache_tensors=[], kv_cache_groups=[group])


def test_starkv_cache_manager_ingest_and_take_plans():
    config = _minimal_kv_cache_config()
    manager = StarKVCacheManager(
        kv_cache_config=config,
        max_model_len=256,
        enable_caching=False,
    )

    out = ModelRunnerOutput(
        req_ids=["a", "b"],
        req_id_to_index={"a": 0, "b": 1},
        sampled_token_ids=[[1], [2]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
        starkv_feedback={
            "a": {
                "request_id": "a",
                "confidence": 0.5,
                "start_pos": 2,
                "end_pos": 5,
                "suffix_len": 3,
            },
        },
    )
    manager.ingest_feedback(out)
    plans = manager.take_reforward_plans()
    assert len(plans) == 1
    assert plans[0]["request_id"] == "a"
    assert plans[0]["suffix_len"] == 3

    stats = manager.get_starkv_stats()
    assert stats.total_reforward_requests == 1
    assert stats.total_reforward_tokens == 3


def test_starkv_cache_manager_take_empty_without_ingest():
    config = _minimal_kv_cache_config()
    manager = StarKVCacheManager(
        kv_cache_config=config,
        max_model_len=256,
        enable_caching=False,
    )
    plans = manager.take_reforward_plans()
    assert plans == []
    assert manager.get_starkv_stats().total_reforward_requests == 0


def test_starkv_stats_to_dict():
    s = StarKVStats(total_reforward_requests=2, total_reforward_tokens=10)
    d = s.to_dict()
    assert d["total_reforward_requests"] == 2
    assert d["total_reforward_tokens"] == 10


def test_starkv_cache_manager_reset_stats():
    config = _minimal_kv_cache_config()
    manager = StarKVCacheManager(
        kv_cache_config=config,
        max_model_len=256,
        enable_caching=False,
    )
    out = ModelRunnerOutput(
        req_ids=["x"],
        req_id_to_index={"x": 0},
        sampled_token_ids=[[1]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
        starkv_feedback={"x": {"request_id": "x", "suffix_len": 1}},
    )
    manager.ingest_feedback(out)
    manager.take_reforward_plans()
    assert manager.get_starkv_stats().total_reforward_requests == 1
    manager.reset_starkv_stats()
    assert manager.get_starkv_stats().total_reforward_requests == 0
