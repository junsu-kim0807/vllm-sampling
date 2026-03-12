# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.starkv.offload import StarKVOffloadStore


def test_offload_store_offload_and_restore():
    store = StarKVOffloadStore()
    store.offload("req1", [1, 2, 3])
    store.offload("req1", [4])
    restored = store.restore("req1")
    assert restored == [1, 2, 3, 4]
    assert store.restore("req1") == []


def test_offload_store_clear_request():
    store = StarKVOffloadStore()
    store.offload("req1", [1, 2])
    store.clear_request("req1")
    assert store.restore("req1") == []
