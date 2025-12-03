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


def test_adapter_raises_without_dependency(monkeypatch):
    monkeypatch.delitem(sys.modules, "starkv", raising=False)
    cache_config = CacheConfig()
    cache_config.enable_starkv_super_cache = True

    with pytest.raises(RuntimeError):
        StarkVPressAdapter(cache_config)

