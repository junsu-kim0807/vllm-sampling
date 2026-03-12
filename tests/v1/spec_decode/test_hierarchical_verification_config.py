# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for hierarchical verification config and spec decode cost breakdown metrics.

Run from repo root:
  pytest tests/v1/spec_decode/test_hierarchical_verification_config.py -v
  python scripts/run_hierarchical_verification_tests.py
"""

import pytest
from pydantic import ValidationError

from vllm.config import ModelConfig, ParallelConfig, SpeculativeConfig
from vllm.v1.spec_decode.metrics import SpecDecodingLogging, SpecDecodingStats


def _make_ngram_speculative_config(
    hierarchical_verification: bool = False,
    compress_method: str = "random",
    compression_ratio: float = 0.5,
) -> SpeculativeConfig:
    """Minimal SpeculativeConfig with ngram (no draft model) for config tests."""
    target_model_config = ModelConfig(
        "facebook/opt-125m",
        trust_remote_code=True,
    )
    return SpeculativeConfig(
        method="ngram",
        prompt_lookup_min=1,
        prompt_lookup_max=1,
        num_speculative_tokens=2,
        target_model_config=target_model_config,
        target_parallel_config=ParallelConfig(),
        hierarchical_verification=hierarchical_verification,
        compress_method=compress_method,
        compression_ratio=compression_ratio,
    )


class TestHierarchicalVerificationConfig:
    """Tests for hierarchical_verification, compress_method, compression_ratio."""

    def test_defaults(self) -> None:
        """Without hierarchical_verification, defaults are applied."""
        config = _make_ngram_speculative_config()
        assert config.hierarchical_verification is False
        assert config.compress_method == "random"
        assert config.compression_ratio == 0.5

    def test_hierarchical_verification_explicit(self) -> None:
        """Explicit hierarchical_verification=True and custom compression."""
        config = _make_ngram_speculative_config(
            hierarchical_verification=True,
            compress_method="random",
            compression_ratio=0.25,
        )
        assert config.hierarchical_verification is True
        assert config.compress_method == "random"
        assert config.compression_ratio == 0.25

    def test_compression_ratio_bounds(self) -> None:
        """compression_ratio must be in (0, 1]."""
        with pytest.raises(ValidationError):
            _make_ngram_speculative_config(compression_ratio=0.0)
        with pytest.raises(ValidationError):
            _make_ngram_speculative_config(compression_ratio=1.5)
        # valid
        c = _make_ngram_speculative_config(compression_ratio=1.0)
        assert c.compression_ratio == 1.0

    def test_compute_hash_includes_hierarchical_verification(self) -> None:
        """Hash differs when hierarchical_verification or compression params change."""
        c1 = _make_ngram_speculative_config(hierarchical_verification=False)
        c2 = _make_ngram_speculative_config(hierarchical_verification=True)
        assert c1.compute_hash() != c2.compute_hash()

        c3 = _make_ngram_speculative_config(
            hierarchical_verification=True, compression_ratio=0.3
        )
        c4 = _make_ngram_speculative_config(
            hierarchical_verification=True, compression_ratio=0.5
        )
        assert c3.compute_hash() != c4.compute_hash()


class TestSpecDecodingStatsCostBreakdown:
    """Tests for SpecDecodingStats cost breakdown fields and observe_draft_with_cost_breakdown."""

    def test_new_has_zero_cost_breakdown(self) -> None:
        stats = SpecDecodingStats.new(num_spec_tokens=4)
        assert stats.draft_time_sec == 0.0
        assert stats.compression_time_sec == 0.0
        assert stats.partial_verification_time_sec == 0.0
        assert stats.num_partial_accepted_tokens == 0
        assert stats.full_verification_time_sec == 0.0

    def test_observe_draft_unchanged(self) -> None:
        """observe_draft still works and does not touch cost breakdown."""
        stats = SpecDecodingStats.new(num_spec_tokens=4)
        stats.observe_draft(num_draft_tokens=4, num_accepted_tokens=2)
        assert stats.num_drafts == 1
        assert stats.num_draft_tokens == 4
        assert stats.num_accepted_tokens == 2
        assert stats.draft_time_sec == 0.0
        assert stats.full_verification_time_sec == 0.0

    def test_observe_draft_with_cost_breakdown(self) -> None:
        """observe_draft_with_cost_breakdown accumulates draft + cost breakdown."""
        stats = SpecDecodingStats.new(num_spec_tokens=4)
        stats.observe_draft_with_cost_breakdown(
            num_draft_tokens=4,
            num_accepted_tokens=2,
            draft_time_sec=0.01,
            compression_time_sec=0.002,
            partial_verification_time_sec=0.005,
            num_partial_accepted_tokens=3,
            full_verification_time_sec=0.008,
        )
        assert stats.num_drafts == 1
        assert stats.num_draft_tokens == 4
        assert stats.num_accepted_tokens == 2
        assert stats.draft_time_sec == 0.01
        assert stats.compression_time_sec == 0.002
        assert stats.partial_verification_time_sec == 0.005
        assert stats.num_partial_accepted_tokens == 3
        assert stats.full_verification_time_sec == 0.008

        # second call accumulates
        stats.observe_draft_with_cost_breakdown(
            num_draft_tokens=4,
            num_accepted_tokens=1,
            draft_time_sec=0.02,
            full_verification_time_sec=0.01,
        )
        assert stats.num_drafts == 2
        assert stats.draft_time_sec == pytest.approx(0.03)
        assert stats.full_verification_time_sec == pytest.approx(0.018)
        # partial left unchanged (not passed)
        assert stats.num_partial_accepted_tokens == 3

    def test_observe_draft_with_cost_breakdown_defaults(self) -> None:
        """Cost breakdown kwargs default to 0 so observe_draft is equivalent."""
        stats = SpecDecodingStats.new(num_spec_tokens=2)
        stats.observe_draft_with_cost_breakdown(
            num_draft_tokens=2,
            num_accepted_tokens=1,
        )
        assert stats.num_drafts == 1
        assert stats.num_accepted_tokens == 1
        assert stats.draft_time_sec == 0.0
        assert stats.compression_time_sec == 0.0
        assert stats.partial_verification_time_sec == 0.0
        assert stats.num_partial_accepted_tokens == 0
        assert stats.full_verification_time_sec == 0.0


class TestSpecDecodingLoggingCostBreakdown:
    """Tests for SpecDecodingLogging with cost breakdown fields."""

    def test_observe_and_reset_includes_cost_breakdown(self) -> None:
        logging = SpecDecodingLogging()
        stats = SpecDecodingStats.new(num_spec_tokens=2)
        stats.observe_draft_with_cost_breakdown(
            num_draft_tokens=2,
            num_accepted_tokens=1,
            draft_time_sec=0.01,
            full_verification_time_sec=0.02,
        )
        logging.observe(stats)
        assert len(logging.draft_time_sec) == 1
        assert logging.draft_time_sec[0] == 0.01
        assert len(logging.full_verification_time_sec) == 1
        assert logging.full_verification_time_sec[0] == 0.02

        logging.log(log_fn=lambda *args, **kwargs: None)  # no-op log
        logging.reset()
        assert len(logging.draft_time_sec) == 0
        assert len(logging.full_verification_time_sec) == 0
