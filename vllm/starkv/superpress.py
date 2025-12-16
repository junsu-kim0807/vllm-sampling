# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Internal SuperPress implementation for StarKV.
This is a placeholder implementation that allows testing StarKV integration
without requiring an external starkv package.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from vllm.logger import init_logger

logger = init_logger(__name__)


class SuperPress:
    """
    Internal SuperPress implementation for StarKV.
    
    This is a simplified placeholder implementation that provides basic
    compression scoring functionality. For production use, a full-featured
    SuperPress implementation should be provided via the external 'starkv' package.
    """

    def __init__(self):
        self.compression_ratio: float | None = None
        """Target compression ratio (0.0 to 1.0)."""
        self.score_fn: str | None = None
        """Score function identifier (e.g., 'morphkv', 'kvzip')."""
        self.confidence_threshold: float | None = None
        """Confidence threshold for determining low-confidence requests."""
        self.last_payload: Dict[str, Any] | None = None
        """Last payload processed (for debugging)."""

    def score_prefill(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score a prefill snapshot and return compression decisions.
        
        Args:
            payload: Dictionary containing:
                - layer_name: str
                - kv_cache_group_id: int
                - request_ids: list[str]
                - num_tokens: int
                - slot_mapping: Any
                - token_request_indices: list[int]
                - token_positions: list[int]
        
        Returns:
            Dictionary with:
                - confidence_by_request: dict[str, float]
                - keep_tokens_by_request: dict[str, list[int]] (optional)
                - low_confidence_requests: list[str] (optional)
        """
        self.last_payload = payload
        
        request_ids = payload.get("request_ids", [])
        num_tokens = payload.get("num_tokens", 0)
        
        # Default confidence calculation.
        #
        # IMPORTANT: In vLLM's current placeholder StarKV plumbing, marking
        # requests as low-confidence can trigger reforward scheduling, which can
        # lead to stalls if the signal is always-on. Therefore the internal
        # SuperPress defaults to "high confidence" unless explicitly overridden
        # for testing.
        base_confidence = float(os.getenv("STARKV_INTERNAL_CONFIDENCE", "1.0"))
        base_confidence = max(0.0, min(1.0, base_confidence))
        
        # Generate confidence scores for each request
        confidence_by_request = {
            rid: base_confidence for rid in request_ids
        }
        
        # Determine which tokens to keep based on compression ratio
        # IMPORTANT:
        # Returning keep_tokens_by_request triggers vLLM's current placeholder
        # block-drop/offload plumbing (see GPUModelRunner._apply_starkv_keep_plan).
        # Without a real SuperPress that understands vLLM's KV/block mapping, this
        # can corrupt state or crash. Therefore, internal SuperPress only emits
        # keep plans when explicitly requested.
        keep_tokens_by_request = None
        enable_keep_plan = bool(int(os.getenv("STARKV_INTERNAL_ENABLE_KEEP_PLAN", "0")))
        if enable_keep_plan and self.compression_ratio is not None and num_tokens > 0:
            # Keep a fraction of tokens based on compression ratio.
            num_keep = max(1, int(num_tokens * (1.0 - self.compression_ratio)))
            keep_indices = [int(i * num_tokens / num_keep) for i in range(num_keep)]
            keep_tokens_by_request = {rid: keep_indices for rid in request_ids}
        
        # Determine low confidence requests
        low_confidence_requests = []
        if self.confidence_threshold is not None:
            low_confidence_requests = [
                rid
                for rid, conf in confidence_by_request.items()
                if conf < self.confidence_threshold
            ]
        
        result = {
            "confidence_by_request": confidence_by_request,
        }
        
        if keep_tokens_by_request is not None:
            result["keep_tokens_by_request"] = keep_tokens_by_request
        
        if low_confidence_requests:
            result["low_confidence_requests"] = low_confidence_requests
        
        logger.debug(
            "SuperPress scored %d requests, %d tokens. "
            "Compression ratio: %s, Confidence threshold: %s",
            len(request_ids),
            num_tokens,
            self.compression_ratio,
            self.confidence_threshold,
        )
        
        return result

