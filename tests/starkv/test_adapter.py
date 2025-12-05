# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types

import pytest
import torch

from vllm.config.cache import CacheConfig
from vllm.starkv.adapter import (
    StarKVLayerSnapshot,
    StarKVPressAdapter,
)


@pytest.fixture
def fake_starkv_module(monkeypatch):
    module = types.SimpleNamespace()

    class FakeSuperPress:
        def __init__(self):
            self.compression_ratio = None
            self.score_fn = None
            self.confidence_threshold = None
            self.last_payload = None

        def score_prefill(self, payload):
            self.last_payload = payload
            request_ids = payload["request_ids"]
            return {
                "confidence_by_request": {rid: 0.6 for rid in request_ids},
                "keep_tokens_by_request": {rid: [0] for rid in request_ids},
                "debug": {"num_tokens": payload["num_tokens"]},
            }

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

    adapter = StarKVPressAdapter(cache_config)
    assert adapter.press is not None
    assert adapter.is_available
    assert adapter.press.compression_ratio == 0.4
    assert adapter.press.score_fn == "morphkv"
    assert adapter.press.confidence_threshold == 0.8


def test_adapter_handles_missing_dependency(monkeypatch):
    monkeypatch.delitem(sys.modules, "starkv", raising=False)
    cache_config = CacheConfig()
    cache_config.enable_starkv_super_cache = True

    adapter = StarKVPressAdapter(cache_config)
    assert adapter.press is None
    assert not adapter.is_available


def test_adapter_scores_snapshot_with_press(fake_starkv_module):
    cache_config = CacheConfig()
    cache_config.enable_starkv_super_cache = True
    adapter = StarKVPressAdapter(cache_config)
    snapshot = StarKVLayerSnapshot(
        layer_name="layer0",
        kv_cache_group_id=0,
        request_ids=["req0", "req1"],
        num_tokens=4,
        slot_mapping=torch.zeros((2, 2), dtype=torch.int64),
        token_request_indices=[0, 0, 1, 1],
        token_positions=[0, 1, 0, 1],
    )
    result = adapter.process_prefill_snapshot(snapshot)
    assert result is not None
    assert result.keep_tokens_by_request == {"req0": [0], "req1": [0]}
    assert adapter.press.last_payload["layer_name"] == "layer0"


def test_adapter_falls_back_when_press_lacks_score(fake_starkv_module):
    cache_config = CacheConfig()
    cache_config.enable_starkv_super_cache = True
    adapter = StarKVPressAdapter(cache_config)
    assert adapter.press is not None
    press_cls = type(adapter.press)
    if hasattr(press_cls, "score_prefill"):
        delattr(press_cls, "score_prefill")

    snapshot = StarKVLayerSnapshot(
        layer_name="layer0",
        kv_cache_group_id=0,
        request_ids=["req0"],
        num_tokens=2,
        slot_mapping=torch.ones((1, 1), dtype=torch.int64),
        token_request_indices=[0, 0],
        token_positions=[0, 1],
    )
    result = adapter.process_prefill_snapshot(snapshot)
    assert result is not None
    assert result.confidence_by_request["req0"] == 1.0
    assert result.keep_tokens_by_request is None

