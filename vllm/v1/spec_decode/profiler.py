from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import socket
import time
from pathlib import Path
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.spec_decode.profiler_types import (
    PendingCudaEventSpan,
    SpecDecodeBatchMetadataRecord,
    SpecDecodeCostBreakdownRecord,
    SpecDecodeFamilyMetadataRecord,
    SpecDecodeProfileMode,
    SpecDecodeProfileTimingBackend,
    SpecDecodeProfileTransport,
    SpecDecodeRequestMetadataRecord,
)

logger = init_logger(__name__)


class JsonlProfileWriter:
    FILE_COST = "spec_decode_cost_breakdown"
    FILE_WORKER = "spec_decode_worker_batch_metadata"
    FILE_SCHED = "spec_decode_scheduler_request_metadata"
    FILE_FAMILY = "spec_decode_family_metadata"

    def __init__(
        self,
        output_dir: str | None,
        profile_writer_role: str,
        profile_writer_id: str,
        profile_writer_pid: int,
        profile_writer_host: str,
        flush_interval: int,
        mode: SpecDecodeProfileMode,
        timing_backend: SpecDecodeProfileTimingBackend,
        request_sample_rate: float,
        max_reqs_per_step_record: int | None,
        include_token_ids: bool,
    ) -> None:
        self.output_dir = Path(output_dir) if output_dir else None
        self.profile_writer_role = profile_writer_role
        self.profile_writer_id = profile_writer_id
        self.profile_writer_pid = profile_writer_pid
        self.profile_writer_host = profile_writer_host
        self.flush_interval = max(1, flush_interval)
        self._buffers: dict[str, list[dict[str, Any]]] = {}
        self._buffered_rows = 0
        self._emitted_files: set[str] = set()
        self._closed = False
        self._disabled_due_to_error = False
        self._warned_error = False
        self._manifest_path: Path | None = None
        self._manifest: dict[str, Any] = {
            "schema_version": 1,
            "profile_mode": mode.value,
            "timing_backend": timing_backend.value,
            "request_sample_rate": request_sample_rate,
            "max_reqs_per_step_record": max_reqs_per_step_record,
            "include_token_ids": include_token_ids,
            "profile_writer_role": profile_writer_role,
            "profile_writer_id": profile_writer_id,
            "profile_writer_pid": profile_writer_pid,
            "profile_writer_host": profile_writer_host,
            "time_created": time.time(),
            "time_closed": None,
            "emitted_files": [],
        }

        if self.output_dir is not None:
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self._handle_io_failure("mkdir", e)
                return
            self._manifest_path = (
                self.output_dir
                / f"spec_decode_profile_manifest.{self.profile_writer_id}.json"
            )
            try:
                self._flush_manifest()
            except Exception as e:
                self._handle_io_failure("manifest_init", e)

    def _filename_for_kind(self, kind: str) -> str:
        return f"{kind}.{self.profile_writer_id}.jsonl"

    def _append(self, kind: str, record: dict[str, Any]) -> None:
        if self.output_dir is None or self._disabled_due_to_error:
            return
        self._buffers.setdefault(kind, []).append(record)
        self._buffered_rows += 1
        self._emitted_files.add(self._filename_for_kind(kind))
        if self._buffered_rows >= self.flush_interval:
            self.flush()

    def write_cost(self, record: SpecDecodeCostBreakdownRecord) -> None:
        self._append(self.FILE_COST, dataclasses.asdict(record))

    def write_worker_batch_metadata(self, record: SpecDecodeBatchMetadataRecord) -> None:
        self._append(self.FILE_WORKER, dataclasses.asdict(record))

    def write_scheduler_request_metadata(
        self, record: SpecDecodeRequestMetadataRecord
    ) -> None:
        self._append(self.FILE_SCHED, dataclasses.asdict(record))

    def write_family_metadata(self, record: SpecDecodeFamilyMetadataRecord) -> None:
        self._append(self.FILE_FAMILY, dataclasses.asdict(record))

    def flush(self) -> None:
        if self.output_dir is None or self._disabled_due_to_error:
            return
        try:
            for kind, rows in list(self._buffers.items()):
                if not rows:
                    continue
                file_path = self.output_dir / self._filename_for_kind(kind)
                with file_path.open("a", encoding="utf-8") as f:
                    for row in rows:
                        f.write(json.dumps(row, sort_keys=True) + "\n")
                self._buffers[kind] = []
            self._buffered_rows = 0
            self._manifest["emitted_files"] = sorted(self._emitted_files)
            self._flush_manifest()
        except Exception as e:
            self._handle_io_failure("flush", e)

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        self._manifest["time_closed"] = time.time()
        try:
            self._flush_manifest()
        except Exception as e:
            self._handle_io_failure("close", e)
        self._closed = True

    def _flush_manifest(self) -> None:
        if self._manifest_path is None or self._disabled_due_to_error:
            return
        with self._manifest_path.open("w", encoding="utf-8") as f:
            json.dump(self._manifest, f, indent=2, sort_keys=True)

    def _handle_io_failure(self, stage: str, error: Exception) -> None:
        # Best-effort profiling: never let profiling I/O break inference.
        self._disabled_due_to_error = True
        self._buffers.clear()
        self._buffered_rows = 0
        if not self._warned_error:
            logger.warning(
                "Spec decode profiler disabled after %s I/O failure for writer %s: %s",
                stage,
                self.profile_writer_id,
                error,
            )
            self._warned_error = True


class SpecDecodeProfiler:
    """Unified speculative profiler.

    Assumes at most one active speculative profiling step per writer.
    """

    def __init__(
        self,
        mode: SpecDecodeProfileMode,
        timing_backend: SpecDecodeProfileTimingBackend,
        writer: JsonlProfileWriter | None,
        include_token_ids: bool,
        request_sample_rate: float,
        max_reqs_per_step_record: int | None,
        max_steps: int | None,
        profile_writer_role: str,
        profile_writer_id: str,
        profile_writer_pid: int,
        profile_writer_host: str,
    ) -> None:
        self.mode = mode
        self.timing_backend = timing_backend
        self.writer = writer
        self.include_token_ids = include_token_ids
        self.request_sample_rate = request_sample_rate
        self.max_reqs_per_step_record = max_reqs_per_step_record
        self.max_steps = max_steps
        self.profile_writer_role = profile_writer_role
        self.profile_writer_id = profile_writer_id
        self.profile_writer_pid = profile_writer_pid
        self.profile_writer_host = profile_writer_host
        self._active_step_id: int | None = None
        self._recorded_steps = 0
        self._spans: dict[str, PendingCudaEventSpan] = {}
        self._request_records_in_step = 0

    @property
    def cost_enabled(self) -> bool:
        return self.mode.cost_enabled

    @property
    def metadata_enabled(self) -> bool:
        return self.mode.metadata_enabled

    @property
    def active_spec_profile_step_id(self) -> int | None:
        """Writer-local step id for the current profiling step, if any."""
        return self._active_step_id

    def begin_step(self, spec_profile_step_id: int) -> None:
        if self.mode == SpecDecodeProfileMode.DISABLED:
            return
        # Strict max_steps: do not activate a new step once the cap is reached,
        # so emits and timing for subsequent steps stay disabled.
        if self.max_steps is not None and self._recorded_steps >= self.max_steps:
            return
        self._drain_unfinished_spans()
        self._active_step_id = spec_profile_step_id
        self._spans.clear()
        self._request_records_in_step = 0

    def start_stage(self, stage_name: str) -> None:
        if not self.cost_enabled or self._active_step_id is None:
            return
        span = PendingCudaEventSpan(stage_name=stage_name)
        if self.timing_backend == SpecDecodeProfileTimingBackend.CUDA_EVENT:
            span.start_event = torch.cuda.Event(enable_timing=True)
            span.end_event = torch.cuda.Event(enable_timing=True)
            span.start_event.record()
        else:
            torch.cuda.synchronize()
            span.start_perf_counter = time.perf_counter()
        self._spans[stage_name] = span

    def end_stage(self, stage_name: str) -> None:
        if not self.cost_enabled or self._active_step_id is None:
            return
        span = self._spans.get(stage_name)
        if span is None:
            return
        if self.timing_backend == SpecDecodeProfileTimingBackend.CUDA_EVENT:
            assert span.end_event is not None
            span.end_event.record()
        else:
            torch.cuda.synchronize()
            assert span.start_perf_counter is not None
            span.elapsed_ms = (time.perf_counter() - span.start_perf_counter) * 1000.0
            span.completed = True

    def add_stage_elapsed_ms(self, stage_name: str, elapsed_ms: float) -> None:
        if not self.cost_enabled or self._active_step_id is None:
            return
        span = self._spans.get(stage_name)
        if span is None:
            span = PendingCudaEventSpan(stage_name=stage_name)
            self._spans[stage_name] = span
        span.elapsed_ms = (span.elapsed_ms or 0.0) + elapsed_ms
        span.completed = True

    def emit_worker_batch_metadata(
        self, record: SpecDecodeBatchMetadataRecord
    ) -> None:
        if not self.metadata_enabled or self.writer is None:
            return
        if self.max_steps is not None and self._recorded_steps >= self.max_steps:
            return
        self.writer.write_worker_batch_metadata(record)

    def emit_scheduler_request_metadata(
        self, record: SpecDecodeRequestMetadataRecord
    ) -> None:
        if not self.metadata_enabled or self.writer is None:
            return
        if self.max_steps is not None and self._recorded_steps >= self.max_steps:
            return
        if self.max_reqs_per_step_record is not None:
            if self._request_records_in_step >= self.max_reqs_per_step_record:
                return
        if not self._should_sample_request(record.spec_profile_step_id, record.req_id):
            return
        self._request_records_in_step += 1
        self.writer.write_scheduler_request_metadata(record)

    def emit_family_metadata(self, record: SpecDecodeFamilyMetadataRecord) -> None:
        if not self.metadata_enabled or self.writer is None:
            return
        if self.max_steps is not None and self._recorded_steps >= self.max_steps:
            return
        self.writer.write_family_metadata(record)

    def finalize_step_transport(
        self,
        num_partial_accepted_per_req: list[int] | None = None,
        *,
        num_intermediate_accepted_per_req: list[int] | None = None,
        num_intermediate_verified_tokens_per_req: list[int] | None = None,
        staged_verification_depth: int | None = None,
        staged_verification_used: bool | None = None,
    ) -> SpecDecodeProfileTransport | None:
        if self.mode == SpecDecodeProfileMode.DISABLED or self._active_step_id is None:
            return None
        if self.max_steps is not None and self._recorded_steps >= self.max_steps:
            self._active_step_id = None
            self._spans.clear()
            return None

        cost_record: SpecDecodeCostBreakdownRecord | None = None
        if self.cost_enabled:
            cost_record = SpecDecodeCostBreakdownRecord(
                profile_writer_role=self.profile_writer_role,
                profile_writer_id=self.profile_writer_id,
                profile_writer_pid=self.profile_writer_pid,
                profile_writer_host=self.profile_writer_host,
                spec_profile_step_id=self._active_step_id,
                target_forward_time_ms=self._finalize_stage_elapsed_ms(
                    "target_forward"
                ),
                draft_forward_time_ms=self._finalize_stage_elapsed_ms("draft_forward"),
                reject_sample_time_ms=self._finalize_stage_elapsed_ms("reject_sample"),
                bookkeeping_time_ms=self._finalize_stage_elapsed_ms("bookkeeping"),
                partial_verification_time_ms=self._finalize_stage_elapsed_ms(
                    "partial_verification"
                ),
                intermediate_verification_time_ms=self._finalize_stage_elapsed_ms(
                    "intermediate_verification"
                ),
                num_partial_accepted_per_req=num_partial_accepted_per_req,
                num_intermediate_accepted_per_req=num_intermediate_accepted_per_req,
                num_intermediate_verified_tokens_per_req=(
                    num_intermediate_verified_tokens_per_req
                ),
                staged_verification_depth=staged_verification_depth,
                staged_verification_used=staged_verification_used,
            )
            if self.writer is not None:
                self.writer.write_cost(cost_record)

        transport = SpecDecodeProfileTransport(
            spec_profile_step_id=self._active_step_id,
            cost_breakdown=cost_record,
        )
        self._recorded_steps += 1
        self._active_step_id = None
        self._spans.clear()
        return transport

    def close(self) -> None:
        self._drain_unfinished_spans()
        if self.writer is not None:
            self.writer.close()

    def req_ids_hash(self, req_ids: list[str]) -> str:
        key = "|".join(req_ids)
        return hashlib.sha1(key.encode("utf-8"), usedforsecurity=False).hexdigest()

    def _finalize_stage_elapsed_ms(self, stage_name: str) -> float:
        span = self._spans.get(stage_name)
        if span is None:
            return 0.0
        if span.completed and span.elapsed_ms is not None:
            return span.elapsed_ms
        if self.timing_backend == SpecDecodeProfileTimingBackend.CUDA_EVENT:
            if span.start_event is None or span.end_event is None:
                return 0.0
            span.end_event.synchronize()
            span.elapsed_ms = float(span.start_event.elapsed_time(span.end_event))
            span.completed = True
            return span.elapsed_ms
        return span.elapsed_ms or 0.0

    def _drain_unfinished_spans(self) -> None:
        for name in list(self._spans.keys()):
            try:
                self._finalize_stage_elapsed_ms(name)
            except Exception:
                logger.debug("Failed to finalize speculative profile span %s", name)

    def _should_sample_request(self, step_id: int, req_id: str) -> bool:
        if self.request_sample_rate >= 1.0:
            return True
        if self.request_sample_rate <= 0.0:
            return False
        key = f"{self.profile_writer_id}|{step_id}|{req_id}"
        digest = hashlib.sha1(key.encode("utf-8"), usedforsecurity=False).hexdigest()
        value = int(digest[:8], 16) / float(0xFFFFFFFF)
        return value <= self.request_sample_rate


class NullSpecDecodeProfiler(SpecDecodeProfiler):
    def __init__(self) -> None:
        super().__init__(
            mode=SpecDecodeProfileMode.DISABLED,
            timing_backend=SpecDecodeProfileTimingBackend.CUDA_EVENT,
            writer=None,
            include_token_ids=False,
            request_sample_rate=1.0,
            max_reqs_per_step_record=None,
            max_steps=None,
            profile_writer_role="none",
            profile_writer_id="none",
            profile_writer_pid=os.getpid(),
            profile_writer_host=socket.gethostname(),
        )


def _infer_writer_id(vllm_config: VllmConfig, role: str) -> str:
    parallel_config = vllm_config.parallel_config
    if role == "scheduler":
        return f"scheduler.engine{parallel_config.data_parallel_index}"
    pp_rank = int(os.environ.get("VLLM_PP_RANK", "0"))
    tp_rank = int(os.environ.get("VLLM_TP_RANK", "0"))
    return (
        f"worker.dp{parallel_config.data_parallel_rank}."
        f"pp{pp_rank}."
        f"tp{tp_rank}"
    )


def create_spec_decode_profiler(
    vllm_config: VllmConfig,
    role: str,
) -> SpecDecodeProfiler:
    obs = vllm_config.observability_config
    mode = SpecDecodeProfileMode(obs.spec_decode_profile_mode)
    if mode == SpecDecodeProfileMode.DISABLED:
        return NullSpecDecodeProfiler()
    timing_backend = SpecDecodeProfileTimingBackend(
        obs.spec_decode_profile_timing_backend
    )
    profile_writer_id = _infer_writer_id(vllm_config, role)
    writer = JsonlProfileWriter(
        output_dir=obs.spec_decode_profile_output_dir,
        profile_writer_role=role,
        profile_writer_id=profile_writer_id,
        profile_writer_pid=os.getpid(),
        profile_writer_host=socket.gethostname(),
        flush_interval=obs.spec_decode_profile_flush_interval,
        mode=mode,
        timing_backend=timing_backend,
        request_sample_rate=obs.spec_decode_profile_request_sample_rate,
        max_reqs_per_step_record=obs.spec_decode_profile_max_reqs_per_step_record,
        include_token_ids=obs.spec_decode_profile_include_token_ids,
    )
    return SpecDecodeProfiler(
        mode=mode,
        timing_backend=timing_backend,
        writer=writer,
        include_token_ids=obs.spec_decode_profile_include_token_ids,
        request_sample_rate=obs.spec_decode_profile_request_sample_rate,
        max_reqs_per_step_record=obs.spec_decode_profile_max_reqs_per_step_record,
        max_steps=obs.spec_decode_profile_max_steps,
        profile_writer_role=role,
        profile_writer_id=profile_writer_id,
        profile_writer_pid=os.getpid(),
        profile_writer_host=socket.gethostname(),
    )
