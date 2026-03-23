# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for spec decode cost breakdown metrics (SpecDecodingStats / logging)."""

import pytest

from vllm.v1.spec_decode.metrics import SpecDecodingLogging, SpecDecodingStats


class TestSpecDecodingStatsCostBreakdown:
    """Tests for SpecDecodingStats cost breakdown and observe_draft_with_cost_breakdown."""

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
