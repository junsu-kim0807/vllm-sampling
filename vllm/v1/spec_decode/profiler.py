# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified speculative decode profiler — two-layer design.

Layer A (Online): always-on lightweight profiler for stage cost, shape/memory
    snapshots, and KV-load / KV-working-set proxies.
Layer B (Deep): sampled-step kernel-level profiler using torch.profiler raw
    CUDA events with GEMM memory/compute split and confidence tags.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator

from vllm.logger import init_logger

from .profiler_types import (
    GEMM_KERNEL_PATTERNS,
    KERNEL_CATEGORY_MAP_V1,
    KERNEL_MAP_VERSION,
    KernelConfidence,
    PendingCudaEventSpan,
    SpecDecodeBatchMetadataRecord,
    SpecDecodeFamilyMetadataRecord,
    SpecDecodeProfileMode,
    SpecDecodeProfileTimingBackend,
    SpecDecodeProfileTransport,
    SpecDecodeRequestMetadataRecord,
    SpecDecodeStageKernelRecord,
    SpecDecodeStageMemoryRecord,
    SpecDecodeStageShapeRecord,
    SpecDecodeStageCostRecord,
    StepProfileContext,
    _MemorySnapshotState,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# JSONL writer
# ---------------------------------------------------------------------------

class JsonlProfileWriter:
    """Buffered JSONL writer — one file per record kind.

    IO failures are best-effort: they log a warning but never break inference.
    """

    def __init__(self, output_dir: str, writer_id: str,
                 flush_interval: int = 64):
        self._output_dir = Path(output_dir)
        self._writer_id = writer_id
        self._flush_interval = flush_interval
        self._buffers: dict[str, list[str]] = {}
        self._counts: dict[str, int] = {}
        self._closed = False

        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.warning("profiler: cannot create output dir %s",
                           output_dir, exc_info=True)

    def append(self, kind: str, record: Any) -> None:
        if self._closed:
            return
        buf = self._buffers.setdefault(kind, [])
        try:
            buf.append(json.dumps(asdict(record), default=str))
        except (TypeError, ValueError):
            logger.warning("profiler: failed to serialize %s record",
                           kind, exc_info=True)
            return
        count = self._counts.get(kind, 0) + 1
        self._counts[kind] = count
        if count % self._flush_interval == 0:
            self._flush_kind(kind)

    def _flush_kind(self, kind: str) -> None:
        buf = self._buffers.get(kind)
        if not buf:
            return
        fname = (f"spec_decode_{kind}.{self._writer_id}.jsonl")
        path = self._output_dir / fname
        try:
            with open(path, "a") as f:
                for line in buf:
                    f.write(line)
                    f.write("\n")
        except OSError:
            logger.warning("profiler: failed to write %s", path,
                           exc_info=True)
        buf.clear()

    def flush_all(self) -> None:
        for kind in list(self._buffers):
            self._flush_kind(kind)

    def write_manifest(self, extra: dict[str, Any] | None = None) -> None:
        manifest: dict[str, Any] = {
            "writer_id": self._writer_id,
            "files": {k: f"spec_decode_{k}.{self._writer_id}.jsonl"
                      for k in self._counts},
            "record_counts": dict(self._counts),
        }
        if extra:
            manifest.update(extra)
        path = (self._output_dir /
                f"spec_decode_profile_manifest.{self._writer_id}.json")
        try:
            with open(path, "w") as f:
                json.dump(manifest, f, indent=2, default=str)
        except OSError:
            logger.warning("profiler: failed to write manifest %s",
                           path, exc_info=True)

    def close(self) -> None:
        self.flush_all()
        self.write_manifest()
        self._closed = True


# ---------------------------------------------------------------------------
# Online profiler (Layer A)
# ---------------------------------------------------------------------------

class OnlineSpecDecodeProfiler:
    """Low-overhead online profiler: stage cost, shape/memory, KV proxies."""

    def __init__(
        self,
        mode: SpecDecodeProfileMode,
        timing_backend: SpecDecodeProfileTimingBackend,
        writer: JsonlProfileWriter,
        writer_id: str,
        max_steps: int | None = None,
        request_sample_rate: float = 1.0,
        max_reqs_per_step: int | None = None,
        include_token_ids: bool = False,
        emit_scheduler_bridge: bool = True,
    ):
        self.mode = mode
        self.timing_backend = timing_backend
        self._writer = writer
        self._writer_id = writer_id
        self._max_steps = max_steps
        self._request_sample_rate = request_sample_rate
        self._max_reqs_per_step = max_reqs_per_step
        self._include_token_ids = include_token_ids
        self._emit_scheduler_bridge = emit_scheduler_bridge
        self._steps_emitted = 0
        self._ctx: StepProfileContext | None = None
        self._pending_spans: dict[str, PendingCudaEventSpan] = {}

    # -- step lifecycle -----------------------------------------------------

    def begin_step(self, step_id: int,
                   forward_reason: str = "") -> StepProfileContext:
        ctx = StepProfileContext(step_id=step_id,
                                forward_reason=forward_reason)
        self._ctx = ctx
        self._pending_spans.clear()
        return ctx

    def finalize_step(
        self,
        ctx: StepProfileContext,
    ) -> SpecDecodeProfileTransport | None:
        if self._max_steps is not None and \
                self._steps_emitted >= self._max_steps:
            self._ctx = None
            return None

        cost_record = self._build_cost_record(ctx)
        stage_times: dict[str, float] = {}
        for name, times in ctx.stage_timings.items():
            stage_times[name] = sum(times)
        transport = SpecDecodeProfileTransport(
            step_id=ctx.step_id,
            cost_record=cost_record,
            stage_times_ms=stage_times,
        )

        if self.mode.cost_enabled and cost_record is not None:
            self._writer.append("stage_cost", cost_record)
        for shape_rec in ctx.stage_shapes:
            self._writer.append("stage_shape", shape_rec)
        for mem_rec in ctx.stage_memories:
            self._writer.append("stage_memory", mem_rec)
        for kern_rec in ctx.kernel_records:
            self._writer.append("stage_kernel", kern_rec)

        self._steps_emitted += 1
        self._ctx = None
        return transport

    def _build_cost_record(
        self, ctx: StepProfileContext
    ) -> SpecDecodeStageCostRecord | None:
        if not self.mode.cost_enabled:
            return None

        total_ms = 0.0
        for times in ctx.stage_timings.values():
            total_ms += sum(times)

        rec = SpecDecodeStageCostRecord(
            profile_writer_id=self._writer_id,
            step_id=ctx.step_id,
            stage_name="step_total",
            stage_invocation_idx=0,
            forward_reason=ctx.forward_reason,
            wall_time_ms=total_ms,
            num_intermediate_verified_tokens_per_req=(
                ctx.inter_verified_per_req),
            num_intermediate_accepted_tokens_per_req=(
                ctx.inter_accepted_per_req),
            num_partial_accepted_per_req=ctx.partial_accepted_per_req,
            staged_verification_depth=ctx.staged_verification_depth,
            staged_verification_used=ctx.staged_verification_used,
            bookkeeping_metadata_build_ms=(
                ctx.bookkeeping_metadata_build_ms),
            bookkeeping_scheduler_bridge_ms=(
                ctx.bookkeeping_scheduler_bridge_ms),
            bookkeeping_output_pack_ms=ctx.bookkeeping_output_pack_ms,
        )
        return rec

    # -- stage timing -------------------------------------------------------

    def start_stage(self, name: str, invocation_idx: int = 0) -> None:
        if not self.mode.cost_enabled or self._ctx is None:
            return

        if self.timing_backend == SpecDecodeProfileTimingBackend.CUDA_EVENT:
            import torch
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            span = PendingCudaEventSpan(
                stage_name=name,
                invocation_idx=invocation_idx,
                start_event=start_event,
                end_event=end_event,
                start_perf_counter=time.perf_counter(),
            )
        else:
            import torch
            torch.cuda.synchronize()
            span = PendingCudaEventSpan(
                stage_name=name,
                invocation_idx=invocation_idx,
                start_perf_counter=time.perf_counter(),
            )
        key = f"{name}:{invocation_idx}"
        self._pending_spans[key] = span

    def end_stage(self, name: str, invocation_idx: int = 0) -> float:
        if not self.mode.cost_enabled or self._ctx is None:
            return 0.0

        key = f"{name}:{invocation_idx}"
        span = self._pending_spans.pop(key, None)
        if span is None:
            return 0.0

        if self.timing_backend == SpecDecodeProfileTimingBackend.CUDA_EVENT:
            span.end_event.record()
            import torch
            torch.cuda.synchronize()
            elapsed_ms = span.start_event.elapsed_time(span.end_event)
        else:
            import torch
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() -
                          span.start_perf_counter) * 1000.0

        timings = self._ctx.stage_timings.setdefault(name, [])
        timings.append(elapsed_ms)

        per_stage_rec = SpecDecodeStageCostRecord(
            profile_writer_id=self._writer_id,
            step_id=self._ctx.step_id,
            stage_name=name,
            stage_invocation_idx=invocation_idx,
            forward_reason=self._ctx.forward_reason,
            wall_time_ms=elapsed_ms,
        )
        self._writer.append("stage_cost", per_stage_rec)
        return elapsed_ms

    def add_stage_elapsed_ms(self, name: str, elapsed_ms: float,
                             invocation_idx: int = 0) -> None:
        if not self.mode.cost_enabled or self._ctx is None:
            return
        timings = self._ctx.stage_timings.setdefault(name, [])
        timings.append(elapsed_ms)

    # -- shape snapshots ----------------------------------------------------

    def snapshot_shape(self, stage_name: str,
                       metadata: dict[str, Any],
                       invocation_idx: int = 0) -> None:
        if not self.mode.shape_memory_enabled or self._ctx is None:
            return

        rec = SpecDecodeStageShapeRecord(
            profile_writer_id=self._writer_id,
            step_id=self._ctx.step_id,
            stage_name=stage_name,
            stage_invocation_idx=invocation_idx,
            forward_reason=self._ctx.forward_reason,
            **{k: v for k, v in metadata.items()
               if hasattr(SpecDecodeStageShapeRecord, k)},
        )
        self._ctx.stage_shapes.append(rec)

    # -- memory snapshots ---------------------------------------------------

    def snapshot_memory_before(self, stage_name: str,
                               invocation_idx: int = 0) -> None:
        if not self.mode.shape_memory_enabled or self._ctx is None:
            return
        import torch
        torch.cuda.reset_peak_memory_stats()
        state = _MemorySnapshotState(
            alloc_before=torch.cuda.memory_allocated(),
            reserved_before=torch.cuda.memory_reserved(),
        )
        key = f"{stage_name}:{invocation_idx}"
        self._ctx._pending_memory_snapshots[key] = state

    def snapshot_memory_after(self, stage_name: str,
                              invocation_idx: int = 0) -> None:
        if not self.mode.shape_memory_enabled or self._ctx is None:
            return
        import torch
        key = f"{stage_name}:{invocation_idx}"
        state = self._ctx._pending_memory_snapshots.pop(key, None)
        if state is None:
            return

        rec = SpecDecodeStageMemoryRecord(
            profile_writer_id=self._writer_id,
            step_id=self._ctx.step_id,
            stage_name=stage_name,
            stage_invocation_idx=invocation_idx,
            alloc_before_bytes=state.alloc_before,
            alloc_after_bytes=torch.cuda.memory_allocated(),
            alloc_peak_bytes=torch.cuda.max_memory_allocated(),
            reserved_before_bytes=state.reserved_before,
            reserved_after_bytes=torch.cuda.memory_reserved(),
            reserved_peak_bytes=torch.cuda.max_memory_reserved(),
        )
        self._ctx.stage_memories.append(rec)

    # -- KV proxies ---------------------------------------------------------

    def record_kv_proxies(
        self,
        stage_name: str,
        *,
        q_lens: list[int] | None = None,
        kv_lens: list[int] | None = None,
        block_table: Any = None,
        page_size: int = 16,
        invocation_idx: int = 0,
    ) -> None:
        """Record KV-load work proxy and KV working-set proxy.

        kv_load_work_proxy = sum(q_len * kv_len) per batch row.
        kv_working_set_pages = distinct pages in block_table for active rows.
        """
        if not self.mode.shape_memory_enabled or self._ctx is None:
            return

        kv_work = 0
        if q_lens is not None and kv_lens is not None:
            for q, k in zip(q_lens, kv_lens):
                kv_work += q * k

        working_set_pages = 0
        live_pages_total = 0
        if block_table is not None:
            try:
                import torch
                if isinstance(block_table, torch.Tensor) and \
                        block_table.numel() > 0:
                    flat = block_table.reshape(-1)
                    valid = flat[flat >= 0]
                    live_pages_total = int(valid.numel())
                    working_set_pages = int(valid.unique().numel())
            except Exception:
                pass

        if self._ctx.stage_shapes:
            for rec in reversed(self._ctx.stage_shapes):
                if (rec.stage_name == stage_name and
                        rec.stage_invocation_idx == invocation_idx):
                    rec.kv_load_work_proxy = kv_work
                    rec.kv_working_set_pages = working_set_pages
                    rec.kv_live_pages_total = live_pages_total
                    break

    # -- metadata records ---------------------------------------------------

    def emit_batch_metadata(self, record: SpecDecodeBatchMetadataRecord
                            ) -> None:
        if not self.mode.metadata_enabled:
            return
        record.profile_writer_id = self._writer_id
        self._writer.append("batch_metadata", record)

    def should_sample_request(self, req_id: str) -> bool:
        """Determine if a request should be included based on sample rate."""
        if self._request_sample_rate >= 1.0:
            return True
        h = int(hashlib.md5(
            f"{self._writer_id}:{req_id}".encode(),
            usedforsecurity=False,
        ).hexdigest()[:8], 16)
        return (h % max(1, int(1.0 / self._request_sample_rate))) == 0

    def emit_request_metadata(self, record: SpecDecodeRequestMetadataRecord
                              ) -> None:
        if not self.mode.metadata_enabled:
            return
        if not self.should_sample_request(record.req_id):
            return
        record.profile_writer_id = self._writer_id
        self._writer.append("request_metadata", record)

    def emit_family_metadata(self, record: SpecDecodeFamilyMetadataRecord
                             ) -> None:
        if not self.mode.metadata_enabled:
            return
        record.profile_writer_id = self._writer_id
        self._writer.append("family_metadata", record)

    # -- close --------------------------------------------------------------

    def close(self) -> None:
        self._writer.close()


# ---------------------------------------------------------------------------
# Deep kernel profiler (Layer B)
# ---------------------------------------------------------------------------

class DeepSpecDecodeKernelProfiler:
    """Sampled-step kernel-level profiler using raw CUDA events."""

    def __init__(
        self,
        writer_id: str,
        sample_rate: float = 0.01,
        gemm_ai_threshold: float = 50.0,
        stage_filter: list[str] | None = None,
    ):
        self._writer_id = writer_id
        self._sample_denominator = max(1, int(1.0 / sample_rate)) \
            if sample_rate > 0 else 0
        self._gemm_ai_threshold = gemm_ai_threshold
        self._stage_filter = set(stage_filter) if stage_filter else None

    def should_profile_step(self, step_id: int) -> bool:
        if self._sample_denominator <= 0:
            return False
        h = int(hashlib.md5(
            f"{self._writer_id}:{step_id}".encode(),
            usedforsecurity=False,
        ).hexdigest()[:8], 16)
        return (h % self._sample_denominator) == 0

    def should_profile_stage(self, stage_name: str) -> bool:
        if self._stage_filter is None:
            return True
        return stage_name in self._stage_filter

    @contextmanager
    def profile_stage(
        self,
        stage_name: str,
        invocation_idx: int = 0,
        *,
        ctx: StepProfileContext | None = None,
        origin_batch_size: int = 0,
        effective_batch_size: int = 0,
        num_tokens: int = 0,
        sum_seq_lens: int = 0,
        max_seq_len: int = 0,
    ) -> Generator[None, None, None]:
        """Context manager that profiles a stage scope via torch.profiler."""
        import torch
        from torch.profiler import ProfilerActivity

        wall_start = time.perf_counter()
        with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
        ) as prof:
            yield

        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        record = self._classify_events(
            prof, stage_name, invocation_idx,
            wall_time_ms=wall_ms,
            ctx=ctx,
            origin_batch_size=origin_batch_size,
            effective_batch_size=effective_batch_size,
            num_tokens=num_tokens,
            sum_seq_lens=sum_seq_lens,
            max_seq_len=max_seq_len,
        )
        if ctx is not None:
            ctx.kernel_records.append(record)

    def _classify_events(
        self,
        prof: Any,
        stage_name: str,
        invocation_idx: int,
        *,
        wall_time_ms: float,
        ctx: StepProfileContext | None,
        origin_batch_size: int,
        effective_batch_size: int,
        num_tokens: int,
        sum_seq_lens: int,
        max_seq_len: int,
    ) -> SpecDecodeStageKernelRecord:
        buckets: dict[str, float] = {
            "kv_sensitive_attention": 0.0,
            "attention_other": 0.0,
            "gemm_memory_dominated": 0.0,
            "gemm_compute_dominated": 0.0,
            "other_compute": 0.0,
            "communication": 0.0,
            "sampling": 0.0,
            "host_overhead": 0.0,
        }
        confidence: dict[str, str] = {
            "kv_sensitive_attention": KernelConfidence.PROXY.value,
            "gemm_memory": KernelConfidence.PROXY.value,
            "gemm_compute": KernelConfidence.PROXY.value,
            "communication": KernelConfidence.PROXY.value,
        }

        try:
            events = prof.events()
        except Exception:
            events = []

        for event in events:
            cuda_time = getattr(event, "self_cuda_time_total", 0) or 0
            if cuda_time <= 0:
                cpu_time = getattr(event, "self_cpu_time_total", 0) or 0
                if cpu_time > 0 and not self._is_cuda_event(event):
                    buckets["host_overhead"] += cpu_time / 1000.0
                continue

            time_ms = cuda_time / 1000.0
            name_lower = getattr(event, "name", "").lower()
            cat, conf = self._classify_single_kernel(
                name_lower, event, time_ms)
            buckets[cat] += time_ms

            if cat == "kv_sensitive_attention":
                confidence["kv_sensitive_attention"] = conf
            elif cat == "gemm_memory_dominated":
                confidence["gemm_memory"] = conf
            elif cat == "gemm_compute_dominated":
                confidence["gemm_compute"] = conf
            elif cat == "communication":
                confidence["communication"] = conf

        categorized_total = sum(buckets.values())
        unattributed = wall_time_ms - categorized_total

        step_id = ctx.step_id if ctx else 0
        forward_reason = ctx.forward_reason if ctx else ""

        return SpecDecodeStageKernelRecord(
            profile_writer_id=self._writer_id,
            step_id=step_id,
            stage_name=stage_name,
            stage_invocation_idx=invocation_idx,
            forward_reason=forward_reason,
            origin_batch_size=origin_batch_size,
            effective_batch_size=effective_batch_size,
            num_tokens=num_tokens,
            sum_seq_lens=sum_seq_lens,
            max_seq_len=max_seq_len,
            wall_time_ms=wall_time_ms,
            kv_sensitive_attention_time_ms=(
                buckets["kv_sensitive_attention"]),
            kv_sensitive_attention_confidence=(
                confidence["kv_sensitive_attention"]),
            attention_other_time_ms=buckets["attention_other"],
            gemm_memory_dominated_time_ms=(
                buckets["gemm_memory_dominated"]),
            gemm_memory_confidence=confidence["gemm_memory"],
            gemm_compute_dominated_time_ms=(
                buckets["gemm_compute_dominated"]),
            gemm_compute_confidence=confidence["gemm_compute"],
            other_compute_time_ms=buckets["other_compute"],
            communication_time_ms=buckets["communication"],
            communication_confidence=confidence["communication"],
            sampling_time_ms=buckets["sampling"],
            host_overhead_time_ms=buckets["host_overhead"],
            unattributed_time_ms=unattributed,
            kernel_map_version=KERNEL_MAP_VERSION,
        )

    def _classify_single_kernel(
        self, name_lower: str, event: Any, time_ms: float
    ) -> tuple[str, str]:
        """Classify a single CUDA kernel -> (bucket_name, confidence)."""
        # Direct classification
        for cat_key, patterns in KERNEL_CATEGORY_MAP_V1.items():
            for pat in patterns:
                if pat in name_lower:
                    if cat_key.startswith("attention"):
                        return ("kv_sensitive_attention",
                                KernelConfidence.DIRECT.value)
                    if cat_key.startswith("communication"):
                        return ("communication",
                                KernelConfidence.DIRECT.value)
                    if cat_key.startswith("sampling"):
                        return ("sampling",
                                KernelConfidence.DIRECT.value)
                    if cat_key.startswith("normalization"):
                        return ("other_compute",
                                KernelConfidence.DIRECT.value)
                    if cat_key.startswith("activation"):
                        return ("other_compute",
                                KernelConfidence.DIRECT.value)

        # GEMM classification
        is_gemm = any(p in name_lower for p in GEMM_KERNEL_PATTERNS)
        if is_gemm:
            stack = getattr(event, "stack", None) or []
            stack_str = " ".join(str(s) for s in stack).lower()

            if any(kw in stack_str for kw in (
                    "attention", "attn", "self_attn")):
                return ("kv_sensitive_attention",
                        KernelConfidence.DERIVED.value)
            if any(kw in stack_str for kw in ("mlp", "ffn", "feed_forward")):
                ai = self._estimate_arithmetic_intensity(event)
                if ai is not None and ai < self._gemm_ai_threshold:
                    return ("gemm_memory_dominated",
                            KernelConfidence.DERIVED.value)
                return ("gemm_compute_dominated",
                        KernelConfidence.DERIVED.value)

            ai = self._estimate_arithmetic_intensity(event)
            if ai is not None:
                if ai < self._gemm_ai_threshold:
                    return ("gemm_memory_dominated",
                            KernelConfidence.PROXY.value)
                return ("gemm_compute_dominated",
                        KernelConfidence.PROXY.value)

            return ("other_compute", KernelConfidence.PROXY.value)

        return ("other_compute", KernelConfidence.PROXY.value)

    @staticmethod
    def _estimate_arithmetic_intensity(event: Any) -> float | None:
        """Estimate FLOPs/bytes for a GEMM from input shapes."""
        shapes = getattr(event, "input_shapes", None)
        if not shapes or len(shapes) < 2:
            return None
        try:
            a_shape = shapes[0]
            b_shape = shapes[1]
            if len(a_shape) < 2 or len(b_shape) < 2:
                return None
            m, k = a_shape[-2], a_shape[-1]
            n = b_shape[-1]
            flops = 2.0 * m * k * n
            bytes_accessed = (m * k + k * n + m * n) * 2  # fp16
            if bytes_accessed == 0:
                return None
            return flops / bytes_accessed
        except (IndexError, TypeError):
            return None

    @staticmethod
    def _is_cuda_event(event: Any) -> bool:
        device_type = getattr(event, "device_type", None)
        if device_type is not None:
            return str(device_type).lower().endswith("cuda")
        return False


# ---------------------------------------------------------------------------
# Null profiler (disabled mode)
# ---------------------------------------------------------------------------

class NullSpecDecodeProfiler:
    """Complete no-op when profiling is disabled."""

    mode = SpecDecodeProfileMode.DISABLED

    def begin_step(self, step_id: int,
                   forward_reason: str = "") -> StepProfileContext:
        return StepProfileContext(step_id=step_id,
                                 forward_reason=forward_reason)

    def finalize_step(
        self, ctx: StepProfileContext
    ) -> SpecDecodeProfileTransport | None:
        return None

    def start_stage(self, name: str, invocation_idx: int = 0) -> None:
        pass

    def end_stage(self, name: str, invocation_idx: int = 0) -> float:
        return 0.0

    def add_stage_elapsed_ms(self, name: str, elapsed_ms: float,
                             invocation_idx: int = 0) -> None:
        pass

    def snapshot_shape(self, stage_name: str,
                       metadata: dict[str, Any],
                       invocation_idx: int = 0) -> None:
        pass

    def snapshot_memory_before(self, stage_name: str,
                               invocation_idx: int = 0) -> None:
        pass

    def snapshot_memory_after(self, stage_name: str,
                              invocation_idx: int = 0) -> None:
        pass

    def record_kv_proxies(self, stage_name: str, **kwargs: Any) -> None:
        pass

    def emit_batch_metadata(self, record: Any) -> None:
        pass

    def emit_request_metadata(self, record: Any) -> None:
        pass

    def emit_family_metadata(self, record: Any) -> None:
        pass

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_spec_decode_profiler(
    vllm_config: VllmConfig,
    role: str = "worker",
) -> OnlineSpecDecodeProfiler | NullSpecDecodeProfiler:
    """Create the appropriate profiler based on observability config."""
    obs_cfg = vllm_config.observability_config
    if obs_cfg is None:
        return NullSpecDecodeProfiler()

    mode_str = getattr(obs_cfg, "spec_decode_profile_mode", "disabled")
    try:
        mode = SpecDecodeProfileMode(mode_str)
    except ValueError:
        logger.warning("Unknown spec_decode_profile_mode=%r, "
                       "falling back to disabled", mode_str)
        return NullSpecDecodeProfiler()

    if mode == SpecDecodeProfileMode.DISABLED:
        return NullSpecDecodeProfiler()

    output_dir = getattr(obs_cfg, "spec_decode_profile_output_dir",
                         None) or "/tmp/vllm_spec_profile"
    timing_str = getattr(obs_cfg, "spec_decode_profile_timing_backend",
                         "cuda_event")
    try:
        timing_backend = SpecDecodeProfileTimingBackend(timing_str)
    except ValueError:
        timing_backend = SpecDecodeProfileTimingBackend.CUDA_EVENT

    flush_interval = getattr(obs_cfg, "spec_decode_profile_flush_interval",
                             64)
    max_steps = getattr(obs_cfg, "spec_decode_profile_max_steps", None)

    writer_id = f"{role}_{os.getpid()}"
    writer = JsonlProfileWriter(output_dir, writer_id, flush_interval)

    request_sample_rate = getattr(
        obs_cfg, "spec_decode_profile_request_sample_rate", 1.0)
    max_reqs_per_step = getattr(
        obs_cfg, "spec_decode_profile_max_reqs_per_step_record", None)
    include_token_ids = getattr(
        obs_cfg, "spec_decode_profile_include_token_ids", False)
    emit_scheduler_bridge = getattr(
        obs_cfg, "spec_decode_profile_emit_scheduler_bridge", True)

    profiler = OnlineSpecDecodeProfiler(
        mode=mode,
        timing_backend=timing_backend,
        writer=writer,
        writer_id=writer_id,
        max_steps=max_steps,
        request_sample_rate=request_sample_rate,
        max_reqs_per_step=max_reqs_per_step,
        include_token_ids=include_token_ids,
        emit_scheduler_bridge=emit_scheduler_bridge,
    )

    if mode.kernel_enabled:
        kernel_sample_rate = getattr(
            obs_cfg, "spec_decode_profile_kernel_sample_rate", 0.01)
        gemm_ai_threshold = getattr(
            obs_cfg, "spec_decode_profile_gemm_ai_threshold", 50.0)
        stage_filter = getattr(
            obs_cfg, "spec_decode_profile_kernel_stage_filter", None)
        profiler._deep_profiler = DeepSpecDecodeKernelProfiler(
            writer_id=writer_id,
            sample_rate=kernel_sample_rate,
            gemm_ai_threshold=gemm_ai_threshold,
            stage_filter=stage_filter,
        )
    else:
        profiler._deep_profiler = None  # type: ignore[attr-defined]

    return profiler
