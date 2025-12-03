# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.request import Request

logger = init_logger(__name__)


class StarkVCacheManager(KVCacheManager):
    """
    Placeholder StarkV-aware cache manager. Phase 3 introduces the plumbing
    needed to track future SuperCache demotions/promotions without altering the
    underlying KV allocation behaviour yet.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._events: list[dict[str, Any]] = []
        self._prefill_feedback: list[dict[str, Any]] = []
        self._reforward_pending: set[str] = set()
        logger.info(
            "StarkV cache manager enabled (placeholder). "
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

    def _record_event(self, kind: str, request_id: str, tokens: int | None) -> None:
        event = {"kind": kind, "request_id": request_id, "tokens": tokens}
        self._events.append(event)
        logger.debug("StarkV cache placeholder event: %s", event)

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
        logger.debug("StarkV prefill feedback: %s", feedback)
        if result is not None:
            for request_id in result.low_confidence_requests:
                self._reforward_pending.add(request_id)

    def get_prefill_feedback(self) -> list[dict[str, Any]]:
        return list(self._prefill_feedback)

    def mark_reforward_needed(self, request_id: str) -> None:
        self._reforward_pending.add(request_id)

    def consume_reforward_flag(self, request_id: str) -> bool:
        if request_id in self._reforward_pending:
            self._reforward_pending.remove(request_id)
            return True
        return False

