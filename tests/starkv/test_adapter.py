# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types

import pytest

from vllm.config.cache import CacheConfig
from vllm.starkv.adapter import StarkVPressAdapter


@pytest.fixture
def fake_starkv_module(monkeypatch):
    module = types.SimpleNamespace()

    class FakeSuperPress:
        def __init__(self):
            self.compression_ratio = None
            self.score_fn = None
            self.confidence_threshold = None

    module.SuperPress = FakeSuperPress
    monkeypatch.setitem(sys.modules, "starkv", module)
    yield module
    monkeypatch.delitem(sys.modules, "starkv", raising=False)


def test_adapter_initializes_with_fake_module(fake_starkv_module):
    cache_config = CacheConfig()
    cache_config.enable_starkv_super_cache = True
    cache_config.starkv_compression_ratio = 0.4
    cache_config.starkv_score_fn = "morphkv"
    cache_config.starkv_confidence_threshold = 0.8

    adapter = StarkVPressAdapter(cache_config)
    assert adapter.press is not None
    assert adapter.press.compression_ratio == 0.4
    assert adapter.press.score_fn == "morphkv"
    assert adapter.press.confidence_threshold == 0.8


def test_adapter_uses_internal_fallback_without_dependency(monkeypatch):
    """When external starkv is missing, adapter uses internal SuperPress."""
    monkeypatch.delitem(sys.modules, "starkv", raising=False)
    cache_config = CacheConfig()
    cache_config.enable_starkv_super_cache = True

    adapter = StarkVPressAdapter(cache_config)
    assert adapter.press is not None
    assert hasattr(adapter.press, "score_prefill")
    assert hasattr(adapter.press, "score_decode")


def test_adapter_score_prefill_internal_fallback(monkeypatch):
    monkeypatch.delitem(sys.modules, "starkv", raising=False)
    from vllm.starkv.superpress import StarKVLayerSnapshot

    cache_config = CacheConfig()
    cache_config.enable_starkv_super_cache = True
    adapter = StarkVPressAdapter(cache_config)

    snapshot = StarKVLayerSnapshot(
        layer_name="layer0",
        kv_cache_group_id=0,
        request_ids=["req1", "req2"],
        num_tokens=100,
    )
    result = adapter.score_prefill(snapshot)
    assert "req1" in result.confidence_by_request
    assert "req2" in result.confidence_by_request
    assert result.keep_tokens_by_request is None or "req1" in result.keep_tokens_by_request


def test_adapter_score_decode_low_confidence_triggers_plan(monkeypatch):
    monkeypatch.delitem(sys.modules, "starkv", raising=False)
    cache_config = CacheConfig()
    cache_config.enable_starkv_super_cache = True
    cache_config.starkv_confidence_threshold = 0.9
    adapter = StarkVPressAdapter(cache_config)

    result = adapter.score_decode(
        confidence_by_request={"r1": 0.5, "r2": 0.95},
        current_pos_by_request={"r1": 10, "r2": 10},
        last_reforward_index_by_request={},
        prompt_len_by_request={"r1": 2, "r2": 2},
    )
    assert len(result.plans) >= 1
    plan_req_ids = [p["request_id"] for p in result.plans]
    assert "r1" in plan_req_ids
    assert "r2" not in plan_req_ids


def test_adapter_tier_store_mirror_and_clear(monkeypatch):
    monkeypatch.delitem(sys.modules, "starkv", raising=False)
    cache_config = CacheConfig()
    cache_config.enable_starkv_super_cache = True
    adapter = StarkVPressAdapter(cache_config)

    adapter.mirror_blocks("req1", [1, 2, 3])
    adapter.mirror_blocks("req1", [4])
    assert len(adapter.get_tier_store().get_mirrored_blocks("req1")) == 4
    adapter.clear_request("req1")
    assert adapter.get_tier_store().get_mirrored_blocks("req1") == []

