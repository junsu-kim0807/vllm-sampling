# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
StarKV cache manager: wraps KVCacheManager and adds reforward plan ingestion
and StarKV stats (total_reforward_requests, total_reforward_tokens).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.kv_cache_interface import KVCacheConfig

if TYPE_CHECKING:
    from vllm.v1.outputs import ModelRunnerOutput


@dataclass
class StarKVStats:
    total_reforward_requests: int = 0
    total_reforward_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_reforward_requests": self.total_reforward_requests,
            "total_reforward_tokens": self.total_reforward_tokens,
        }


class StarKVCacheManager(KVCacheManager):
    """
    KV cache manager with StarKV reforward support. Delegates all cache
    operations to the base implementation and adds feedback ingestion and
    reforward plan consumption for the scheduler.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        enable_caching: bool = True,
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
        dcp_world_size: int = 1,
    ) -> None:
        super().__init__(
            kv_cache_config=kv_cache_config,
            max_model_len=max_model_len,
            enable_caching=enable_caching,
            use_eagle=use_eagle,
            log_stats=log_stats,
            enable_kv_cache_events=enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
        )
        self._starkv_stats = StarKVStats()
        self._pending_reforward_plans: list[dict[str, Any]] = []

    def ingest_feedback(self, model_runner_output: ModelRunnerOutput) -> None:
        """Store reforward plans from worker output for the scheduler to consume."""
        feedback = getattr(
            model_runner_output, "starkv_feedback", None
        )
        if not feedback or not isinstance(feedback, dict):
            return
        for req_id, plan in feedback.items():
            if isinstance(plan, dict) and "request_id" in plan:
                self._pending_reforward_plans.append(dict(plan))

    def take_reforward_plans(self) -> list[dict[str, Any]]:
        """Return and clear pending reforward plans; update StarKV stats."""
        plans = self._pending_reforward_plans
        self._pending_reforward_plans = []
        for p in plans:
            self._starkv_stats.total_reforward_requests += 1
            self._starkv_stats.total_reforward_tokens += int(
                p.get("suffix_len", 0)
            )
        return plans

    def get_starkv_stats(self) -> StarKVStats:
        return self._starkv_stats

    def reset_starkv_stats(self) -> None:
        self._starkv_stats = StarKVStats()
