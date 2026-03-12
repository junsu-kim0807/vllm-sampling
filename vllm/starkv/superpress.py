from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class StarKVLayerSnapshot:
    layer_name: str
    kv_cache_group_id: int
    request_ids: list[str]
    num_tokens: int
    token_positions: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StarKVPrefillResult:
    confidence_by_request: dict[str, float]
    low_confidence_requests: list[str]
    keep_tokens_by_request: dict[str, list[int]] | None = None


@dataclass
class StarKVReforwardPlan:
    request_id: str
    confidence: float
    start_pos: int
    end_pos: int
    suffix_len: int


@dataclass
class StarKVDecodeResult:
    plans: list[dict[str, Any]]


class SuperPress:
    """
    Internal StarKV fallback implementation.

    This intentionally keeps logic simple and deterministic so that vLLM-core
    integration and metrics can be validated even without an external `starkv`
    package.
    """

    def __init__(self):
        self.compression_ratio: float = 0.5
        self.score_fn: str = "morphkv"
        self.confidence_threshold: float = 0.8
        self.max_reforward_steps: int | None = None

        self._reforward_steps_by_request: dict[str, int] = {}

    def score_prefill(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_ids = list(payload.get("request_ids", []))
        if not request_ids:
            return {
                "confidence_by_request": {},
                "low_confidence_requests": [],
                "keep_tokens_by_request": {},
            }

        num_tokens = int(payload.get("num_tokens", 0))
        keep_count = max(1, int(round(num_tokens * (1.0 - self.compression_ratio)))) if num_tokens > 0 else 0

        base_conf = float(np.clip(1.0 - self.compression_ratio * 0.5, 0.0, 1.0))
        confidence_by_request = {rid: base_conf for rid in request_ids}
        low_conf = [rid for rid, conf in confidence_by_request.items() if conf <= self.confidence_threshold]

        keep_tokens_by_request = None
        if num_tokens > 0:
            keep = list(range(keep_count))
            keep_tokens_by_request = {rid: keep for rid in request_ids}

        return {
            "confidence_by_request": confidence_by_request,
            "low_confidence_requests": low_conf,
            "keep_tokens_by_request": keep_tokens_by_request,
        }

    def score_decode(self, payload: dict[str, Any]) -> dict[str, Any]:
        confidence_by_request = payload.get("confidence_by_request", {}) or {}
        current_pos_by_request = payload.get("current_pos_by_request", {}) or {}
        last_reforward_index_by_request = payload.get("last_reforward_index_by_request", {}) or {}
        prompt_len_by_request = payload.get("prompt_len_by_request", {}) or {}

        plans: list[dict[str, Any]] = []
        for req_id, confidence in confidence_by_request.items():
            confidence_f = float(confidence)
            if confidence_f > self.confidence_threshold:
                continue

            step_count = self._reforward_steps_by_request.get(req_id, 0)
            if self.max_reforward_steps is not None and step_count >= self.max_reforward_steps:
                continue

            end_pos = int(current_pos_by_request.get(req_id, 0))
            prompt_len = int(prompt_len_by_request.get(req_id, 0))
            default_start = max(prompt_len, end_pos - 1)
            start_pos = int(last_reforward_index_by_request.get(req_id, default_start))
            start_pos = max(prompt_len, min(start_pos, end_pos))
            suffix_len = max(0, end_pos - start_pos)
            if suffix_len <= 0:
                continue

            self._reforward_steps_by_request[req_id] = step_count + 1
            plans.append(
                {
                    "request_id": req_id,
                    "confidence": confidence_f,
                    "start_pos": start_pos,
                    "end_pos": end_pos,
                    "suffix_len": suffix_len,
                }
            )

        return {"plans": plans}
