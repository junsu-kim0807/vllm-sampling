# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any

from vllm.logger import init_logger
from vllm.starkv.adapter import StarKVLayerFeedback, StarKVPrefillResult
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.request import Request

logger = init_logger(__name__)


class StarKVCacheManager(KVCacheManager):
    """
    Placeholder StarKV-aware cache manager. Phase 3 introduces the plumbing
    needed to track future SuperCache demotions/promotions without altering the
    underlying KV allocation behaviour yet.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._events: list[dict[str, Any]] = []
        self._prefill_feedback: list[dict[str, Any]] = []
        self._reforward_pending: set[str] = set()
        self._super_cache_tokens: int = 0
        self._offloaded_blocks: dict[str, list[dict[str, Any]]] = {}
        self._pending_restore: set[str] = set()
        self._reforward_plans: dict[str, dict[str, Any]] = {}
        self.offload_enabled = getattr(
            self.kv_cache_config, "starkv_offload", False
        )
        logger.info(
            "StarKV cache manager enabled (placeholder). "
            "Future phases will hook SuperPress decisions here."
        )

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_lookahead_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
    ) -> KVCacheBlocks | None:
        blocks = super().allocate_slots(
            request=request,
            num_new_tokens=num_new_tokens,
            num_new_computed_tokens=num_new_computed_tokens,
            new_computed_blocks=new_computed_blocks,
            num_lookahead_tokens=num_lookahead_tokens,
            delay_cache_blocks=delay_cache_blocks,
            num_encoder_tokens=num_encoder_tokens,
        )
        if blocks is not None:
            self._record_event("allocate", request.request_id, num_new_tokens)
        return blocks

    def free(self, request: Request) -> None:
        super().free(request)
        self._record_event("free", request.request_id, None)
        self._offloaded_blocks.pop(request.request_id, None)
        self._pending_restore.discard(request.request_id)
        self._reforward_plans.pop(request.request_id, None)

    def _record_event(self, kind: str, request_id: str, tokens: int | None) -> None:
        event = {"kind": kind, "request_id": request_id, "tokens": tokens}
        self._events.append(event)
        logger.debug("StarKV cache placeholder event: %s", event)

    def get_placeholder_events(self) -> list[dict[str, Any]]:
        return list(self._events)

    def record_prefill_feedback(
        self,
        layer_name: str,
        request_ids: list[str],
        num_tokens: int,
        result: Any | None = None,
    ) -> None:
        feedback = {
            "layer": layer_name,
            "requests": list(request_ids),
            "tokens": num_tokens,
            "result": result,
        }
        self._prefill_feedback.append(feedback)
        logger.debug("StarKV prefill feedback: %s", feedback)
        if result is not None:
            for request_id in result.low_confidence_requests:
                self._reforward_pending.add(request_id)

    def ingest_layer_feedbacks(
        self, feedbacks: list[StarKVLayerFeedback]
    ) -> None:
        for feedback in feedbacks:
            snapshot = feedback.snapshot
            self.record_prefill_feedback(
                snapshot.layer_name,
                snapshot.request_ids,
                snapshot.num_tokens,
                feedback.result,
            )
            self._apply_scheduler_block_plan(feedback.result)
            self._record_offloaded_blocks(feedback.result)
            self._record_reforward_plans(feedback.result)

    def _apply_scheduler_block_plan(self, result: StarKVPrefillResult) -> None:
        metadata = result.metadata or {}
        plans = metadata.get("starkv_block_plan")
        if not plans:
            return
        for plan in plans:
            req_id = plan.get("request_id")
            group_id = plan.get("kv_group_id")
            dropped_blocks = plan.get("dropped_blocks")
            if (
                req_id is None
                or group_id is None
                or not dropped_blocks
                or group_id >= len(self.coordinator.single_type_managers)
            ):
                continue
            manager = self.coordinator.single_type_managers[group_id]
            req_blocks = manager.req_to_blocks.get(req_id)
            if not req_blocks:
                continue
            drop_set = set(dropped_blocks)
            new_blocks: list[KVCacheBlock] = []
            freed_blocks: list[KVCacheBlock] = []
            for block in req_blocks:
                if block.block_id in drop_set:
                    freed_blocks.append(block)
                else:
                    new_blocks.append(block)
            if not freed_blocks:
                continue
            manager.req_to_blocks[req_id] = new_blocks
            manager.num_cached_block[req_id] = min(
                manager.num_cached_block.get(req_id, 0),
                len(new_blocks),
            )
            self.block_pool.free_blocks(reversed(freed_blocks))

    def get_prefill_feedback(self) -> list[dict[str, Any]]:
        return list(self._prefill_feedback)

    def mark_reforward_needed(self, request_id: str) -> None:
        self._reforward_pending.add(request_id)

    def consume_reforward_flag(self, request_id: str) -> bool:
        if request_id in self._reforward_pending:
            self._reforward_pending.remove(request_id)
            self._pending_restore.add(request_id)
            return True
        return False

    def demote_to_super_cache(self, request_id: str, num_tokens: int) -> None:
        if not self.offload_enabled:
            return
        self._super_cache_tokens += num_tokens
        logger.debug(
            "StarKV offload placeholder: request %s demoted %d tokens "
            "(total super cache=%d)",
            request_id,
            num_tokens,
            self._super_cache_tokens,
        )

    def get_super_cache_usage(self) -> int:
        return self._super_cache_tokens

    def _record_offloaded_blocks(self, result: StarKVPrefillResult) -> None:
        metadata = result.metadata or {}
        entries = metadata.get("starkv_offloaded_blocks")
        if not entries:
            return
        for entry in entries:
            req_id = entry.get("request_id")
            if req_id is None:
                continue
            self._offloaded_blocks.setdefault(req_id, []).append(entry)

    def pop_offloaded_handles(self, request_id: str) -> list[dict[str, Any]] | None:
        if request_id not in self._pending_restore:
            return None
        self._pending_restore.discard(request_id)
        return self._offloaded_blocks.pop(request_id, None)

    def _record_reforward_plans(self, result: StarKVPrefillResult) -> None:
        metadata = result.metadata or {}
        entries = metadata.get("starkv_reforward_requests")
        if not entries:
            return
        for entry in entries:
            req_id = entry.get("request_id")
            if req_id is None:
                continue
            self._reforward_plans[req_id] = entry

    def pop_reforward_plan(self, request_id: str) -> dict[str, Any] | None:
        return self._reforward_plans.pop(request_id, None)

