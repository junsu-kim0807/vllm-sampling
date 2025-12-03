# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import Request
from vllm.v1.core.kv_cache_utils import NONE_HASH, get_request_block_hasher
from vllm.v1.core.starkv_cache_manager import StarkVCacheManager
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec


def _make_request(request_id: str, token_ids: list[int], block_size: int) -> Request:
    hash_fn = lambda data: (str(data).encode("utf-8") if data is not None else NONE_HASH)
    return Request(
        request_id=request_id,
        prompt_token_ids=token_ids,
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
        eos_token_id=100,
        lora_request=None,
        cache_salt=None,
        block_hasher=get_request_block_hasher(block_size, caching_hash_fn=hash_fn),
    )


def _make_config(block_size: int, num_blocks: int) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(block_size, 1, 1, torch.float32),
            )
        ],
        enable_starkv_super_cache=True,
    )


def test_starkv_cache_manager_records_placeholder_events():
    block_size = 16
    manager = StarkVCacheManager(
        kv_cache_config=_make_config(block_size, 8),
        max_model_len=1024,
        enable_caching=True,
        use_eagle=False,
        log_stats=False,
        enable_kv_cache_events=False,
        dcp_world_size=1,
    )

    tokens = list(range(block_size * 2))
    req = _make_request("req", tokens, block_size)
    computed_blocks, num_tokens = manager.get_computed_blocks(req)
    blocks = manager.allocate_slots(
        req,
        num_new_tokens=len(tokens),
        num_new_computed_tokens=num_tokens,
        new_computed_blocks=computed_blocks,
    )
    assert blocks is not None
    events = manager.get_placeholder_events()
    assert any(event["kind"] == "allocate" for event in events)

    manager.free(req)
    events = manager.get_placeholder_events()
    assert any(event["kind"] == "free" for event in events)

