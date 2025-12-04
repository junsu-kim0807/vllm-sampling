# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List

from vllm.config.cache import CacheConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class StarKVPrefillResult:
    low_confidence_requests: list[str]
    confidence_by_request: dict[str, float]
    keep_tokens_by_request: dict[str, list[int]] | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class StarKVLayerSnapshot:
    layer_name: str
    request_ids: list[str]
    num_tokens: int
    slot_mapping: Any  # CPU tensor or list representing slot mapping


class StarKVPressAdapter:
    """
    Thin wrapper around StarKV's SuperPress. Phase 2 only ensures that the
    dependency is imported and configured; actual compression hooks are added
    in later phases.
    """

    def __init__(self, cache_config: CacheConfig):
        self.cache_config = cache_config
        self.force_low_confidence = bool(
            int(os.getenv("STARKV_FORCE_LOW_CONF", "0"))
        )
        self._press: Any | None = None
        self.is_available: bool = False
        self._initialize_press()

    @property
    def press(self) -> Any | None:
        return self._press

    def _initialize_press(self) -> None:
        try:
            from starkv import SuperPress  # type: ignore[attr-defined]
        except ImportError:  # pragma: no cover - exercised via tests
            logger.warning(
                "StarKV SuperCache requested but the 'starkv' package is not "
                "installed. Running without StarKV integration."
            )
            self.is_available = False
            self._press = None
            return

        press = SuperPress()
        self._apply_cache_config(press)
        self._press = press
        self.is_available = True
        logger.info(
            "Initialized StarKV SuperPress (compression_ratio=%s, score_fn=%s, "
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
                "StarKV adapter placeholder invoked but StarKV is unavailable."
            )
        else:
            logger.debug("StarKV adapter placeholder invoked.")

    def process_prefill_snapshot(
        self,
        snapshot: StarKVLayerSnapshot,
    ) -> StarKVPrefillResult | None:
        """
        Forward a prefill snapshot to StarKV's scorer (if available) and convert
        the output into a StarKVPrefillResult consumed by the cache manager.
        """

        if not self.is_available:
            logger.debug(
                "StarKV unavailable; skipping metadata for layer %s (tokens=%d).",
                snapshot.layer_name,
                snapshot.num_tokens,
            )
            return None

        payload = self._build_payload(snapshot)
        logger.debug(
            "StarKV snapshot: layer=%s tokens=%d reqs=%d slots_shape=%s",
            snapshot.layer_name,
            snapshot.num_tokens,
            len(snapshot.request_ids),
            payload["slot_mapping_shape"],
        )

        result = self._score_with_press(snapshot, payload)
        return result

    def _build_payload(self, snapshot: StarKVLayerSnapshot) -> Dict[str, Any]:
        slot_mapping = snapshot.slot_mapping
        slot_shape = None
        if hasattr(slot_mapping, "shape"):
            slot_shape = tuple(slot_mapping.shape)  # type: ignore[arg-type]
        if hasattr(slot_mapping, "numpy"):
            slot_mapping = slot_mapping.numpy()
        elif hasattr(slot_mapping, "tolist"):
            slot_mapping = slot_mapping.tolist()

        return {
            "layer_name": snapshot.layer_name,
            "request_ids": snapshot.request_ids,
            "num_tokens": snapshot.num_tokens,
            "slot_mapping": slot_mapping,
            "slot_mapping_shape": slot_shape,
        }

    def _score_with_press(
        self,
        snapshot: StarKVLayerSnapshot,
        payload: Dict[str, Any],
    ) -> StarKVPrefillResult | None:
        press = self.press
        if press is None or not hasattr(press, "score_prefill"):
            logger.debug(
                "StarKV SuperPress does not expose score_prefill(); "
                "falling back to placeholder confidences."
            )
            return self._placeholder_result(snapshot)

        try:
            raw_result = press.score_prefill(payload)  # type: ignore[attr-defined]
        except Exception:
            logger.exception(
                "StarKV score_prefill() raised; using placeholder confidences."
            )
            return self._placeholder_result(snapshot)

        parsed = self._parse_press_result(snapshot, raw_result)
        if parsed is None:
            logger.warning(
                "StarKV score_prefill() returned an unexpected payload. "
                "Using placeholder confidences."
            )
            return self._placeholder_result(snapshot)
        return parsed

    def _parse_press_result(
        self,
        snapshot: StarKVLayerSnapshot,
        raw_result: Any,
    ) -> StarKVPrefillResult | None:
        if raw_result is None:
            return None
        if isinstance(raw_result, StarKVPrefillResult):
            return raw_result
        if not isinstance(raw_result, dict):
            logger.debug("StarKV score_prefill returned non-dict result: %s", type(raw_result))
            return None

        confidence_map: Dict[str, float] = {}
        raw_conf = raw_result.get("confidence_by_request") or raw_result.get("confidence")
        if isinstance(raw_conf, dict):
            confidence_map = {
                str(req_id): float(score)
                for req_id, score in raw_conf.items()
            }
        elif isinstance(raw_conf, (int, float)):
            confidence_map = {
                rid: float(raw_conf) for rid in snapshot.request_ids
            }
        else:
            confidence_map = {rid: 1.0 for rid in snapshot.request_ids}

        keep_tokens = None
        raw_keep = raw_result.get("keep_tokens_by_request") or raw_result.get(
            "keep_tokens"
        )
        if isinstance(raw_keep, dict):
            keep_tokens = {
                str(req_id): list(map(int, indices))
                for req_id, indices in raw_keep.items()
            }

        threshold = self.cache_config.starkv_confidence_threshold
        low_conf_from_press = raw_result.get("low_confidence_requests")
        if isinstance(low_conf_from_press, list):
            low_conf = [str(rid) for rid in low_conf_from_press]
        elif threshold is not None:
            low_conf = [
                rid
                for rid, score in confidence_map.items()
                if score < threshold
            ]
        else:
            low_conf = []

        metadata = {
            k: v
            for k, v in raw_result.items()
            if k
            not in {
                "confidence_by_request",
                "confidence",
                "keep_tokens_by_request",
                "keep_tokens",
                "low_confidence_requests",
            }
        }

        return StarKVPrefillResult(
            low_confidence_requests=low_conf,
            confidence_by_request=confidence_map,
            keep_tokens_by_request=keep_tokens,
            metadata=metadata or None,
        )

    def _placeholder_result(self, snapshot: StarKVLayerSnapshot) -> StarKVPrefillResult:
        base_confidence = 0.0 if self.force_low_confidence else 1.0
        confidences = {rid: base_confidence for rid in snapshot.request_ids}
        threshold = self.cache_config.starkv_confidence_threshold
        low_conf = []
        if threshold is not None:
            low_conf = [
                rid
                for rid, score in confidences.items()
                if score < threshold
            ]

        return StarKVPrefillResult(
            low_confidence_requests=low_conf,
            confidence_by_request=confidences,
        )

