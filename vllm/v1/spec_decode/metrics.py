# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from dataclasses import dataclass, field

import numpy as np
import prometheus_client

from vllm.config import SpeculativeConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class SpecDecodingStats:
    """Per-step iteration decoding stats from scheduler.

    Each scheduler step, statistics on spec decoding performance are
    aggregated across requests by the scheduler and returned to the
    frontend in EngineCoreOutputs->SchedulerStats.

    Optional cost breakdown (e.g. when profiling staged spec decode):
    - draft_time_sec, compression_time_sec, partial_verification_time_sec,
      full_verification_time_sec: accumulated seconds per step.
    - num_partial_accepted_tokens: optional per-step staged partial-accept counter.
    - full_acceptance_length: use 1 + num_accepted_tokens/num_drafts (existing).
    - end-to-end TPO: use FinishedRequestStats.mean_time_per_output_token.
    """

    num_spec_tokens: int
    num_partial_spec_tokens: int = 0
    num_drafts: int = 0
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0
    num_accepted_tokens_per_pos: list[int] = field(default_factory=list)
    num_partial_accepted_tokens_per_pos: list[int] = field(default_factory=list)

    # Optional cost breakdown timing fields.
    draft_time_sec: float = 0.0
    compression_time_sec: float = 0.0
    partial_verification_time_sec: float = 0.0
    num_partial_accepted_tokens: int = 0
    full_verification_time_sec: float = 0.0
    # Rejection sampling: total time (sec) spent in rejection sampler.
    reject_sample_time_sec: float = 0.0

    # Unified profiler stage times (ms, accumulated per step).
    draft_forward_time_ms: float = 0.0
    target_verify_time_ms: float = 0.0
    intermediate_verify_time_ms: float = 0.0
    expand_collapse_time_ms: float = 0.0

    @classmethod
    def new(
        cls,
        num_spec_tokens: int,
        *,
        num_partial_spec_tokens: int = 0,
    ) -> "SpecDecodingStats":
        return cls(
            num_spec_tokens=num_spec_tokens,
            num_partial_spec_tokens=num_partial_spec_tokens,
            num_accepted_tokens_per_pos=[0] * num_spec_tokens,
            num_partial_accepted_tokens_per_pos=[0] * num_partial_spec_tokens,
        )

    def observe_draft(self, num_draft_tokens: int, num_accepted_tokens: int):
        self.num_drafts += 1
        self.num_draft_tokens += num_draft_tokens
        self.num_accepted_tokens += num_accepted_tokens
        assert num_accepted_tokens <= self.num_spec_tokens
        for i in range(num_accepted_tokens):
            self.num_accepted_tokens_per_pos[i] += 1

    def observe_draft_with_cost_breakdown(
        self,
        num_draft_tokens: int,
        num_accepted_tokens: int,
        *,
        draft_time_sec: float = 0.0,
        compression_time_sec: float = 0.0,
        partial_verification_time_sec: float = 0.0,
        num_partial_accepted_tokens: int = 0,
        full_verification_time_sec: float = 0.0,
        reject_sample_time_sec: float = 0.0,
    ) -> None:
        """Same as observe_draft plus optional cost breakdown fields."""
        self.observe_draft(num_draft_tokens, num_accepted_tokens)
        self.draft_time_sec += draft_time_sec
        self.compression_time_sec += compression_time_sec
        self.partial_verification_time_sec += partial_verification_time_sec
        self.num_partial_accepted_tokens += num_partial_accepted_tokens
        assert num_partial_accepted_tokens <= self.num_partial_spec_tokens
        for i in range(num_partial_accepted_tokens):
            self.num_partial_accepted_tokens_per_pos[i] += 1
        self.full_verification_time_sec += full_verification_time_sec
        self.reject_sample_time_sec += reject_sample_time_sec


class SpecDecodingLogging:
    """Aggregate and log spec decoding metrics.

    LoggingStatLogger aggregates per-iteration metrics over a set
    time interval using observe() and then logs them using log()
    before resetting to zero.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.num_drafts: list[int] = []
        self.num_draft_tokens: list[int] = []
        self.num_accepted_tokens: list[int] = []
        self.accepted_tokens_per_pos_lists: list[list[int]] = []
        self.partial_accepted_tokens_per_pos_lists: list[list[int]] = []
        self.draft_time_sec: list[float] = []
        self.compression_time_sec: list[float] = []
        self.partial_verification_time_sec: list[float] = []
        self.num_partial_accepted_tokens: list[int] = []
        self.full_verification_time_sec: list[float] = []
        self.reject_sample_time_sec_list: list[float] = []
        self.last_log_time = time.monotonic()

    def observe(self, spec_decoding_stats: SpecDecodingStats):
        self.num_drafts.append(spec_decoding_stats.num_drafts)
        self.num_draft_tokens.append(spec_decoding_stats.num_draft_tokens)
        self.num_accepted_tokens.append(spec_decoding_stats.num_accepted_tokens)
        self.accepted_tokens_per_pos_lists.append(
            spec_decoding_stats.num_accepted_tokens_per_pos
        )
        self.partial_accepted_tokens_per_pos_lists.append(
            spec_decoding_stats.num_partial_accepted_tokens_per_pos
        )
        self.draft_time_sec.append(spec_decoding_stats.draft_time_sec)
        self.compression_time_sec.append(spec_decoding_stats.compression_time_sec)
        self.partial_verification_time_sec.append(
            spec_decoding_stats.partial_verification_time_sec
        )
        self.num_partial_accepted_tokens.append(
            spec_decoding_stats.num_partial_accepted_tokens
        )
        self.full_verification_time_sec.append(
            spec_decoding_stats.full_verification_time_sec
        )
        self.reject_sample_time_sec_list.append(
            getattr(spec_decoding_stats, "reject_sample_time_sec", 0.0)
        )

    def log(self, log_fn=logger.info):
        if not self.num_drafts:
            return
        num_drafts = np.sum(self.num_drafts)
        num_draft_tokens = np.sum(self.num_draft_tokens)
        num_accepted_tokens = np.sum(self.num_accepted_tokens)
        draft_throughput = 0
        accepted_throughput = 0

        elapsed_time = time.monotonic() - self.last_log_time
        if elapsed_time > 0:
            draft_throughput = num_draft_tokens / elapsed_time
            accepted_throughput = num_accepted_tokens / elapsed_time

        draft_acceptance_rate = (
            num_accepted_tokens / num_draft_tokens * 100
            if num_draft_tokens > 0
            else float("nan")
        )

        # Conventionally, mean acceptance length includes the bonus token
        mean_acceptance_length = 1 + (num_accepted_tokens / num_drafts)

        pos_matrix = np.array(self.accepted_tokens_per_pos_lists)
        acceptance_rates = np.sum(pos_matrix, axis=0) / num_drafts
        rates_str = ", ".join(f"{p:.3f}" for p in acceptance_rates)
        partial_rates_str = ""
        if self.partial_accepted_tokens_per_pos_lists:
            partial_pos_matrix = np.array(self.partial_accepted_tokens_per_pos_lists)
            if partial_pos_matrix.size > 0:
                partial_rates = np.sum(partial_pos_matrix, axis=0) / num_drafts
                partial_rates_str = ", ".join(f"{p:.3f}" for p in partial_rates)

        # Optional cost breakdown
        total_draft_time = np.sum(self.draft_time_sec)
        total_compression_time = np.sum(self.compression_time_sec)
        total_partial_verification_time = np.sum(self.partial_verification_time_sec)
        total_partial_accepted = np.sum(self.num_partial_accepted_tokens)
        total_full_verification_time = np.sum(self.full_verification_time_sec)
        total_reject_sample_time = np.sum(
            getattr(self, "reject_sample_time_sec_list", [])
        )
        has_cost_breakdown = (
            total_draft_time > 0
            or total_compression_time > 0
            or total_partial_verification_time > 0
            or total_full_verification_time > 0
            or total_reject_sample_time > 0
        )
        partial_acceptance_length = (
            1 + total_partial_accepted / num_drafts if num_drafts > 0 else 0.0
        )

        log_fn(
            "SpecDecoding metrics: "
            "Mean acceptance length: %.2f, "
            "Accepted throughput: %.2f tokens/s, "
            "Drafted throughput: %.2f tokens/s, "
            "Accepted: %d tokens, "
            "Drafted: %d tokens, "
            "Per-position acceptance rate: %s, "
            "Avg Draft acceptance rate: %.1f%%",
            mean_acceptance_length,
            accepted_throughput,
            draft_throughput,
            num_accepted_tokens,
            num_draft_tokens,
            rates_str,
            draft_acceptance_rate,
        )
        if has_cost_breakdown:
            log_fn(
                "SpecDecoding cost breakdown: "
                "draft_time: %.4fs, compression_time: %.4fs, "
                "partial_verification_time: %.4fs, partial_acceptance_length: %.2f, "
                "full_verification_time: %.4fs, full_acceptance_length: %.2f, "
                "reject_sample_time: %.4fs",
                total_draft_time,
                total_compression_time,
                total_partial_verification_time,
                partial_acceptance_length,
                total_full_verification_time,
                mean_acceptance_length,
                total_reject_sample_time,
            )
            if partial_rates_str:
                log_fn(
                    "SpecDecoding intermediate per-position acceptance rate: %s",
                    partial_rates_str,
                )
        self.reset()


class SpecDecodingProm:
    """Record spec decoding metrics in Prometheus.

    The acceptance rate can be calculated using a PromQL query:

      rate(vllm:spec_decode_num_accepted_tokens_total[$interval]) /
      rate(vllm:spec_decode_num_draft_tokens_total[$interval])

    The mean acceptance length (conventionally including bonus tokens)
    can be calculated using:

      1 + (
      rate(vllm:spec_decode_num_accepted_tokens_total[$interval]) /
      rate(vllm:spec_decode_num_drafts[$interval]))

    A per-position acceptance rate vector can be computed using

      vllm:spec_decode_num_accepted_tokens_per_pos[$interval] /
      vllm:spec_decode_num_drafts[$interval]
    """

    _counter_cls = prometheus_client.Counter

    def __init__(
        self,
        speculative_config: SpeculativeConfig | None,
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        self.spec_decoding_enabled = speculative_config is not None
        if not self.spec_decoding_enabled:
            return

        counter_drafts = self._counter_cls(
            name="vllm:spec_decode_num_drafts",
            documentation="Number of spec decoding drafts.",
            labelnames=labelnames,
        )
        self.counter_spec_decode_num_drafts = make_per_engine(
            counter_drafts, per_engine_labelvalues
        )

        counter_draft_tokens = self._counter_cls(
            name="vllm:spec_decode_num_draft_tokens",
            documentation="Number of draft tokens.",
            labelnames=labelnames,
        )
        self.counter_spec_decode_num_draft_tokens = make_per_engine(
            counter_draft_tokens, per_engine_labelvalues
        )

        counter_accepted_tokens = self._counter_cls(
            name="vllm:spec_decode_num_accepted_tokens",
            documentation="Number of accepted tokens.",
            labelnames=labelnames,
        )
        self.counter_spec_decode_num_accepted_tokens = make_per_engine(
            counter_accepted_tokens, per_engine_labelvalues
        )

        assert speculative_config is not None
        num_spec_tokens = (
            speculative_config.runner_num_speculative_tokens()
            if self.spec_decoding_enabled
            else 0
        )
        num_partial_spec_tokens = (
            speculative_config.runner_num_partial_speculative_tokens()
            if self.spec_decoding_enabled
            else 0
        )
        pos_labelnames = labelnames + ["position"]
        base_counter = self._counter_cls(
            name="vllm:spec_decode_num_accepted_tokens_per_pos",
            documentation="Accepted tokens per draft position.",
            labelnames=pos_labelnames,
        )
        self.counter_spec_decode_num_accepted_tokens_per_pos: dict[
            int, list[prometheus_client.Counter]
        ] = {
            idx: [base_counter.labels(*lv, str(pos)) for pos in range(num_spec_tokens)]
            for idx, lv in per_engine_labelvalues.items()
        }
        partial_base_counter = self._counter_cls(
            name="vllm:spec_decode_num_partial_accepted_tokens_per_pos",
            documentation="Accepted intermediate-verification tokens per draft position.",
            labelnames=pos_labelnames,
        )
        self.counter_spec_decode_num_partial_accepted_tokens_per_pos: dict[
            int, list[prometheus_client.Counter]
        ] = {
            idx: [
                partial_base_counter.labels(*lv, str(pos))
                for pos in range(num_partial_spec_tokens)
            ]
            for idx, lv in per_engine_labelvalues.items()
        }

        # Cumulative draft and verification time (seconds). Use Gauge so
        # get_metrics() returns the current cumulative value (works across processes).
        self._cumulative_draft_time_sec: dict[int, float] = {i: 0.0 for i in per_engine_labelvalues}
        self._cumulative_verification_time_sec: dict[int, float] = {
            i: 0.0 for i in per_engine_labelvalues
        }
        gauge_draft_time = prometheus_client.Gauge(
            name="vllm:spec_decode_draft_time_seconds_total",
            documentation="Total draft phase time in seconds.",
            labelnames=labelnames,
        )
        self.gauge_spec_decode_draft_time_seconds = make_per_engine(
            gauge_draft_time, per_engine_labelvalues
        )
        gauge_verification_time = prometheus_client.Gauge(
            name="vllm:spec_decode_verification_time_seconds_total",
            documentation="Total verification phase time in seconds.",
            labelnames=labelnames,
        )
        self.gauge_spec_decode_verification_time_seconds = make_per_engine(
            gauge_verification_time, per_engine_labelvalues
        )
        # Rejection sampling: total time spent in rejection sampler (seconds).
        self._cumulative_reject_sample_time_sec: dict[int, float] = {
            i: 0.0 for i in per_engine_labelvalues
        }
        gauge_reject_sample_time = prometheus_client.Gauge(
            name="vllm:spec_decode_reject_sample_time_seconds_total",
            documentation="Total time spent in rejection sampler (seconds).",
            labelnames=labelnames,
        )
        self.gauge_spec_decode_reject_sample_time = make_per_engine(
            gauge_reject_sample_time, per_engine_labelvalues
        )

        # Unified profiler stage-level cumulative times (ms).
        self._cumul_draft_forward_ms: dict[int, float] = {
            i: 0.0 for i in per_engine_labelvalues}
        self._cumul_target_verify_ms: dict[int, float] = {
            i: 0.0 for i in per_engine_labelvalues}
        self._cumul_intermediate_verify_ms: dict[int, float] = {
            i: 0.0 for i in per_engine_labelvalues}
        self._cumul_expand_collapse_ms: dict[int, float] = {
            i: 0.0 for i in per_engine_labelvalues}
        self.gauge_draft_forward_ms = make_per_engine(
            prometheus_client.Gauge(
                name="vllm:spec_decode_draft_forward_ms_total",
                documentation="Cumulative draft forward time (ms).",
                labelnames=labelnames,
            ), per_engine_labelvalues)
        self.gauge_target_verify_ms = make_per_engine(
            prometheus_client.Gauge(
                name="vllm:spec_decode_target_verify_ms_total",
                documentation="Cumulative target verify time (ms).",
                labelnames=labelnames,
            ), per_engine_labelvalues)
        self.gauge_intermediate_verify_ms = make_per_engine(
            prometheus_client.Gauge(
                name="vllm:spec_decode_intermediate_verify_ms_total",
                documentation="Cumulative intermediate verify time (ms).",
                labelnames=labelnames,
            ), per_engine_labelvalues)
        self.gauge_expand_collapse_ms = make_per_engine(
            prometheus_client.Gauge(
                name="vllm:spec_decode_expand_collapse_ms_total",
                documentation="Cumulative expand/collapse time (ms).",
                labelnames=labelnames,
            ), per_engine_labelvalues)

    def observe(self, spec_decoding_stats: SpecDecodingStats, engine_idx: int = 0):
        if not self.spec_decoding_enabled:
            return
        self.counter_spec_decode_num_drafts[engine_idx].inc(
            spec_decoding_stats.num_drafts
        )
        self.counter_spec_decode_num_draft_tokens[engine_idx].inc(
            spec_decoding_stats.num_draft_tokens
        )
        self.counter_spec_decode_num_accepted_tokens[engine_idx].inc(
            spec_decoding_stats.num_accepted_tokens
        )
        for pos, counter in enumerate(
            self.counter_spec_decode_num_accepted_tokens_per_pos[engine_idx]
        ):
            counter.inc(spec_decoding_stats.num_accepted_tokens_per_pos[pos])
        for pos, counter in enumerate(
            self.counter_spec_decode_num_partial_accepted_tokens_per_pos[engine_idx]
        ):
            counter.inc(spec_decoding_stats.num_partial_accepted_tokens_per_pos[pos])
        # Cumulative times: update Gauges so get_metrics() sees current total.
        if spec_decoding_stats.draft_time_sec > 0:
            self._cumulative_draft_time_sec[engine_idx] += spec_decoding_stats.draft_time_sec
            self.gauge_spec_decode_draft_time_seconds[engine_idx].set(
                self._cumulative_draft_time_sec[engine_idx]
            )
        verification_sec = (
            spec_decoding_stats.partial_verification_time_sec
            + spec_decoding_stats.full_verification_time_sec
        )
        if verification_sec > 0:
            self._cumulative_verification_time_sec[engine_idx] += verification_sec
            self.gauge_spec_decode_verification_time_seconds[engine_idx].set(
                self._cumulative_verification_time_sec[engine_idx]
            )
        reject_time = getattr(spec_decoding_stats, "reject_sample_time_sec", 0.0)
        if reject_time > 0:
            self._cumulative_reject_sample_time_sec[engine_idx] += reject_time
            self.gauge_spec_decode_reject_sample_time[engine_idx].set(
                self._cumulative_reject_sample_time_sec[engine_idx]
            )

        # Unified profiler stage times (ms).
        df = spec_decoding_stats.draft_forward_time_ms
        if df > 0:
            self._cumul_draft_forward_ms[engine_idx] += df
            self.gauge_draft_forward_ms[engine_idx].set(
                self._cumul_draft_forward_ms[engine_idx])
        tv = spec_decoding_stats.target_verify_time_ms
        if tv > 0:
            self._cumul_target_verify_ms[engine_idx] += tv
            self.gauge_target_verify_ms[engine_idx].set(
                self._cumul_target_verify_ms[engine_idx])
        iv = spec_decoding_stats.intermediate_verify_time_ms
        if iv > 0:
            self._cumul_intermediate_verify_ms[engine_idx] += iv
            self.gauge_intermediate_verify_ms[engine_idx].set(
                self._cumul_intermediate_verify_ms[engine_idx])
        ec = spec_decoding_stats.expand_collapse_time_ms
        if ec > 0:
            self._cumul_expand_collapse_ms[engine_idx] += ec
            self.gauge_expand_collapse_ms[engine_idx].set(
                self._cumul_expand_collapse_ms[engine_idx])


def make_per_engine(
    counter: prometheus_client.Counter,
    per_engine_labelvalues: dict[int, list[object]],
):
    """Create a counter for each label value."""
    return {
        idx: counter.labels(*labelvalues)
        for idx, labelvalues in per_engine_labelvalues.items()
    }
