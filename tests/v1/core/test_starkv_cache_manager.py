# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.sampling_params import SamplingParams
from vllm.starkv.adapter import (
    StarKVLayerFeedback,
    StarKVLayerSnapshot,
    StarKVPrefillResult,
)
from vllm.v1.core.kv_cache_manager import Request
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.starkv_cache_manager import StarKVCacheManager
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec


def _make_request(request_id: str, token_ids: list[int], block_size: int) -> Request:
    init_none_hash(lambda seed: str(seed).encode("utf-8"))
    hash_fn = lambda data: str(data).encode("utf-8")
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
    manager = StarKVCacheManager(
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

    result = StarKVPrefillResult(
        low_confidence_requests=["req"], confidence_by_request={"req": 0.5}
    )
    manager.record_prefill_feedback("layer", ["req"], 32, result)
    feedback = manager.get_prefill_feedback()
    assert feedback and feedback[-1]["layer"] == "layer"
    manager.demote_to_super_cache("req", 16)
    if manager.offload_enabled:
        assert manager.get_super_cache_usage() >= 16
    assert manager.consume_reforward_flag("req")


def test_starkv_cache_manager_ingests_feedbacks():
    block_size = 16
    manager = StarKVCacheManager(
        kv_cache_config=_make_config(block_size, 8),
        max_model_len=1024,
        enable_caching=True,
        use_eagle=False,
        log_stats=False,
        enable_kv_cache_events=False,
        dcp_world_size=1,
    )
    tokens = list(range(block_size * 2))
    req = _make_request("reqx", tokens, block_size)
    computed_blocks, num_tokens = manager.get_computed_blocks(req)
    new_blocks = manager.allocate_slots(
        req,
        num_new_tokens=len(tokens),
        num_new_computed_tokens=num_tokens,
        new_computed_blocks=computed_blocks,
    )
    assert new_blocks is not None
    snapshot = StarKVLayerSnapshot(
        layer_name="layer",
        kv_cache_group_id=0,
        request_ids=["reqx"],
        num_tokens=len(tokens),
        slot_mapping=torch.arange(len(tokens), dtype=torch.int64),
        token_request_indices=[0] * len(tokens),
        token_positions=list(range(len(tokens))),
    )
    first_block_id = manager.coordinator.get_blocks("reqx")[0][0].block_id
    result = StarKVPrefillResult(
        low_confidence_requests=["reqx"],
        confidence_by_request={"reqx": 0.6},
        keep_tokens_by_request={"reqx": []},
        metadata={
            "starkv_block_plan": [
                {
                    "request_id": "reqx",
                    "kv_group_id": 0,
                    "dropped_blocks": [first_block_id],
                }
            ]
        },
    )
    feedback = StarKVLayerFeedback(snapshot=snapshot, result=result)
    manager.ingest_layer_feedbacks([feedback])
    blocks = manager.coordinator.get_blocks("reqx")[0]
    assert all(block.block_id != first_block_id for block in blocks)
    assert manager.consume_reforward_flag("reqx")


def test_starkv_cache_manager_ingests_feedbacks():
    manager = StarKVCacheManager(
        kv_cache_config=_make_config(16, 8),
        max_model_len=1024,
        enable_caching=True,
        use_eagle=False,
        log_stats=False,
        enable_kv_cache_events=False,
        dcp_world_size=1,
    )
    snapshot = StarKVLayerSnapshot(
        layer_name="layer0",
        kv_cache_group_id=0,
        request_ids=["r0"],
        num_tokens=8,
        slot_mapping=torch.zeros((1, 1), dtype=torch.int64),
        token_request_indices=[0],
        token_positions=[0],
    )
    result = StarKVPrefillResult(
        low_confidence_requests=["r0"],
        confidence_by_request={"r0": 0.4},
        metadata={
            "starkv_reforward_requests": [
                {"request_id": "r0", "start_pos": 0, "suffix_len": 4}
            ]
        },
    )
    feedback = StarKVLayerFeedback(snapshot=snapshot, result=result)
    manager.ingest_layer_feedbacks([feedback])
    stored = manager.get_prefill_feedback()
    assert stored[-1]["layer"] == "layer0"
    assert manager.consume_reforward_flag("r0")
    plan = manager.pop_reforward_plan("r0")
    assert plan is not None
    assert plan["suffix_len"] == 4

