# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from vllm.config.cache import CacheConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class StarkVPrefillResult:
    low_confidence_requests: list[str]
    confidence_by_request: dict[str, float]


class StarkVPressAdapter:
    """
    Thin wrapper around StarkV's SuperPress. Phase 2 only ensures that the
    dependency is imported and configured; actual compression hooks are added
    in later phases.
    """

    def __init__(self, cache_config: CacheConfig):
        self.cache_config = cache_config
        self._press: Any | None = None
        self.is_available: bool = False
        self.force_low_confidence: bool = False
        self._initialize_press()

    @property
    def press(self) -> Any | None:
        return self._press

    def _initialize_press(self) -> None:
        try:
            from starkv import SuperPress  # type: ignore[attr-defined]
        except ImportError:  # pragma: no cover - exercised via tests
            logger.warning(
                "StarkV SuperCache requested but the 'starkv' package is not "
                "installed. Running without StarkV integration."
            )
            self.is_available = False
            self._press = None
            return

        press = SuperPress()
        self._apply_cache_config(press)
        self._press = press
        self.is_available = True
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

    def log_placeholder_event(self) -> None:
        """
        Temporary helper to make it easy to verify that the adapter is live.
        Later phases will replace this with real compression hooks.
        """

        if not self.is_available:
            logger.debug(
                "StarkV adapter placeholder invoked but StarkV is unavailable."
            )
        else:
            logger.debug("StarkV adapter placeholder invoked.")

    def process_prefill_metadata(
        self,
        layer_name: str,
        num_tokens: int,
        request_ids: list[str],
    ) -> StarkVPrefillResult | None:
        """
        Placeholder bridge that will eventually ship real KV tensors into
        StarKV. For now it simply logs the metadata and emits dummy confidence
        signals to drive the reforward plumbing.
        """

        if not self.is_available:
            logger.debug(
                "StarkV unavailable; skipping metadata for layer %s (tokens=%d).",
                layer_name,
                num_tokens,
            )
            return None

        logger.debug(
            "StarkV metadata: layer=%s tokens=%d reqs=%d",
            layer_name,
            num_tokens,
            len(request_ids),
        )

        base_confidence = 0.0 if self.force_low_confidence else 1.0
        confidences = {rid: base_confidence for rid in request_ids}
        low_conf = []
        threshold = self.cache_config.starkv_confidence_threshold
        if threshold is not None and base_confidence < threshold:
            low_conf = list(request_ids)

        return StarkVPrefillResult(
            low_confidence_requests=low_conf,
            confidence_by_request=confidences,
        )

