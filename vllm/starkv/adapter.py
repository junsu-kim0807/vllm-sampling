# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from vllm.config.cache import CacheConfig
from vllm.logger import init_logger
from vllm.starkv.superpress import (
    StarKVDecodeResult,
    StarKVLayerSnapshot,
    StarKVPrefillResult,
)
from vllm.starkv.tier_store import StarKVTierStore

logger = init_logger(__name__)


class StarkVPressAdapter:
    """
    Thin wrapper around StarkV's SuperPress. Supports external starkv package
    or internal fallback. Provides score_prefill/score_decode and tier mirror.
    """

    def __init__(self, cache_config: CacheConfig):
        self.cache_config = cache_config
        self._press: Any | None = None
        self._tier_store = StarKVTierStore()
        self._initialize_press()

    @property
    def press(self) -> Any | None:
        return self._press

    def _initialize_press(self) -> None:
        try:
            from starkv import SuperPress  # type: ignore[attr-defined]
        except ImportError:
            from vllm.starkv.superpress import SuperPress  # fallback

        press = SuperPress()
        self._apply_cache_config(press)
        self._press = press
        logger.info(
            "Initialized StarkV SuperPress (compression_ratio=%s, score_fn=%s, "
            "confidence_threshold=%s, max_reforward_steps=%s)",
            getattr(press, "compression_ratio", "default"),
            getattr(press, "score_fn", "default"),
            getattr(press, "confidence_threshold", "default"),
            self.cache_config.starkv_max_reforward_steps,
        )

    def _apply_cache_config(self, press: Any) -> None:
        if (
            self.cache_config.starkv_compression_ratio is not None
            and hasattr(press, "compression_ratio")
        ):
            press.compression_ratio = self.cache_config.starkv_compression_ratio

        if (
            self.cache_config.starkv_score_fn is not None
            and hasattr(press, "score_fn")
        ):
            press.score_fn = self.cache_config.starkv_score_fn

        if (
            self.cache_config.starkv_confidence_threshold is not None
            and hasattr(press, "confidence_threshold")
        ):
            press.confidence_threshold = (
                self.cache_config.starkv_confidence_threshold
            )

        if (
            self.cache_config.starkv_max_reforward_steps is not None
            and hasattr(press, "max_reforward_steps")
        ):
            press.max_reforward_steps = self.cache_config.starkv_max_reforward_steps

    def log_placeholder_event(self) -> None:
        """
        Temporary helper to make it easy to verify that the adapter is live.
        Later phases will replace this with real compression hooks.
        """

        if self._press is None:
            logger.debug(
                "StarkV adapter placeholder invoked but SuperPress is not initialized."
            )
        else:
            logger.debug("StarkV adapter placeholder invoked.")

    def score_prefill(self, snapshot: StarKVLayerSnapshot) -> StarKVPrefillResult:
        if self._press is None or not hasattr(self._press, "score_prefill"):
            return StarKVPrefillResult({}, [], None)

        raw = self._press.score_prefill(asdict(snapshot))
        if not isinstance(raw, dict):
            return StarKVPrefillResult({}, [], None)

        confidence_by_request = raw.get("confidence_by_request", {}) or {}
        low_confidence_requests = raw.get("low_confidence_requests", []) or []
        keep_tokens_by_request = raw.get("keep_tokens_by_request")
        return StarKVPrefillResult(
            confidence_by_request={
                str(k): float(v) for k, v in confidence_by_request.items()
            },
            low_confidence_requests=[str(x) for x in low_confidence_requests],
            keep_tokens_by_request=(
                {
                    str(req_id): [int(t) for t in keep]
                    for req_id, keep in keep_tokens_by_request.items()
                }
                if isinstance(keep_tokens_by_request, dict)
                else None
            ),
        )

    def score_decode(
        self,
        *,
        confidence_by_request: dict[str, float],
        current_pos_by_request: dict[str, int],
        last_reforward_index_by_request: dict[str, int],
        prompt_len_by_request: dict[str, int],
    ) -> StarKVDecodeResult:
        if self._press is None or not hasattr(self._press, "score_decode"):
            return StarKVDecodeResult(plans=[])

        payload = {
            "confidence_by_request": confidence_by_request,
            "current_pos_by_request": current_pos_by_request,
            "last_reforward_index_by_request": last_reforward_index_by_request,
            "prompt_len_by_request": prompt_len_by_request,
        }
        raw = self._press.score_decode(payload)
        plans: list[Any] = []
        if isinstance(raw, dict):
            plans = raw.get("plans", []) or []

        parsed = []
        for plan in plans:
            if not isinstance(plan, dict):
                continue
            try:
                parsed.append(
                    {
                        "request_id": str(plan["request_id"]),
                        "confidence": float(plan["confidence"]),
                        "start_pos": int(plan["start_pos"]),
                        "end_pos": int(plan["end_pos"]),
                        "suffix_len": int(plan["suffix_len"]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue

        return StarKVDecodeResult(plans=parsed)

    def get_tier_store(self) -> StarKVTierStore:
        return self._tier_store

    def mirror_blocks(self, request_id: str, block_ids: list[int]) -> None:
        """Record block IDs for sub->super tier sync (mirror) after step."""
        self._tier_store.mirror_blocks(request_id, block_ids)

    def clear_request(self, request_id: str) -> None:
        """Clear tier store state for a finished request."""
        self._tier_store.clear_request(request_id)
