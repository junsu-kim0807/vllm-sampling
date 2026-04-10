# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

from vllm.v1.spec_decode.profiler import JsonlProfileWriter, SpecDecodeProfiler
from vllm.v1.spec_decode.profiler_types import (
    SpecDecodeBatchMetadataRecord,
    SpecDecodeProfileMode,
    SpecDecodeProfileTimingBackend,
    SpecDecodeRequestMetadataRecord,
)
from vllm.v1.spec_decode.spec_stage_runtime import PivotExpansionPlan
from vllm.v1.worker.gpu_model_runner import (
    _collapse_int_sum_by_origin,
    _collapse_winner_int_by_origin,
)


def _build_profiler(
    tmp_path: Path,
    mode: SpecDecodeProfileMode,
    backend: SpecDecodeProfileTimingBackend = SpecDecodeProfileTimingBackend.DEVICE_SYNC,
    max_steps: int | None = None,
) -> SpecDecodeProfiler:
    writer = JsonlProfileWriter(
        output_dir=str(tmp_path),
        profile_writer_role="worker",
        profile_writer_id="worker.dp0.pp0.tp0",
        profile_writer_pid=123,
        profile_writer_host="localhost",
        flush_interval=1,
        mode=mode,
        timing_backend=backend,
        request_sample_rate=1.0,
        max_reqs_per_step_record=None,
        include_token_ids=False,
    )
    return SpecDecodeProfiler(
        mode=mode,
        timing_backend=backend,
        writer=writer,
        include_token_ids=False,
        request_sample_rate=1.0,
        max_reqs_per_step_record=None,
        max_steps=max_steps,
        profile_writer_role="worker",
        profile_writer_id="worker.dp0.pp0.tp0",
        profile_writer_pid=123,
        profile_writer_host="localhost",
    )


def test_disabled_mode_is_noop(tmp_path: Path) -> None:
    profiler = _build_profiler(tmp_path, SpecDecodeProfileMode.DISABLED)
    profiler.begin_step(1)
    profiler.start_stage("target_forward")
    profiler.end_stage("target_forward")
    assert profiler.finalize_step_transport() is None
    profiler.close()
    assert not list(tmp_path.glob("*.jsonl"))


def test_cost_breakdown_emits_transport_and_jsonl(tmp_path: Path) -> None:
    profiler = _build_profiler(tmp_path, SpecDecodeProfileMode.COST_BREAKDOWN)
    profiler.begin_step(7)
    profiler.add_stage_elapsed_ms("target_forward", 3.5)
    profiler.add_stage_elapsed_ms("draft_forward", 1.5)
    profiler.add_stage_elapsed_ms("reject_sample", 0.4)
    profiler.add_stage_elapsed_ms("partial_verification", 0.7)
    profiler.add_stage_elapsed_ms("intermediate_verification", 0.3)
    transport = profiler.finalize_step_transport(
        num_partial_accepted_per_req=[5, 2],
        num_intermediate_accepted_per_req=[1, 0],
        num_intermediate_verified_tokens_per_req=[4, 2],
        staged_verification_depth=3,
        staged_verification_used=True,
    )
    assert transport is not None
    assert transport.spec_profile_step_id == 7
    assert transport.cost_breakdown is not None
    assert transport.cost_breakdown.target_forward_time_ms == 3.5
    assert transport.cost_breakdown.partial_verification_time_ms == 0.7
    assert transport.cost_breakdown.intermediate_verification_time_ms == 0.3
    assert transport.cost_breakdown.num_partial_accepted_per_req == [5, 2]
    assert transport.cost_breakdown.num_intermediate_accepted_per_req == [1, 0]
    assert transport.cost_breakdown.num_intermediate_verified_tokens_per_req == [4, 2]
    assert transport.cost_breakdown.staged_verification_depth == 3
    assert transport.cost_breakdown.staged_verification_used is True
    profiler.close()
    files = list(tmp_path.glob("spec_decode_cost_breakdown.*.jsonl"))
    assert len(files) == 1
    assert files[0].read_text(encoding="utf-8").strip()


def test_metadata_mode_emits_worker_and_scheduler_records(tmp_path: Path) -> None:
    profiler = _build_profiler(tmp_path, SpecDecodeProfileMode.METADATA)
    profiler.begin_step(11)
    profiler.emit_worker_batch_metadata(
        SpecDecodeBatchMetadataRecord(
            profile_writer_role="worker",
            profile_writer_id="worker.dp0.pp0.tp0",
            profile_writer_pid=123,
            profile_writer_host="localhost",
            spec_profile_step_id=11,
            speculative_method="ngram",
            batch_size=2,
            req_ids_hash="hash",
        )
    )
    profiler.emit_scheduler_request_metadata(
        SpecDecodeRequestMetadataRecord(
            profile_writer_role="worker",
            profile_writer_id="worker.dp0.pp0.tp0",
            profile_writer_pid=123,
            profile_writer_host="localhost",
            spec_profile_step_id=11,
            req_id="req-1",
            req_index=0,
            num_draft_tokens=4,
            num_accepted_tokens=2,
            num_rejected_tokens=2,
            num_invalid_spec_tokens=0,
        )
    )
    profiler.close()
    assert list(tmp_path.glob("spec_decode_worker_batch_metadata.*.jsonl"))
    assert list(tmp_path.glob("spec_decode_scheduler_request_metadata.*.jsonl"))


def test_all_mode_combines_cost_and_metadata(tmp_path: Path) -> None:
    profiler = _build_profiler(tmp_path, SpecDecodeProfileMode.ALL)
    profiler.begin_step(21)
    profiler.add_stage_elapsed_ms("target_forward", 9.0)
    profiler.emit_worker_batch_metadata(
        SpecDecodeBatchMetadataRecord(
            profile_writer_role="worker",
            profile_writer_id="worker.dp0.pp0.tp0",
            profile_writer_pid=123,
            profile_writer_host="localhost",
            spec_profile_step_id=21,
            speculative_method="ngram",
            batch_size=1,
            req_ids_hash="hash",
        )
    )
    transport = profiler.finalize_step_transport()
    assert transport is not None
    profiler.close()
    manifest = (
        tmp_path / "spec_decode_profile_manifest.worker.dp0.pp0.tp0.json"
    ).read_text(
        encoding="utf-8"
    )
    assert "profile_mode" in manifest
    assert "time_closed" in manifest


def test_max_steps_blocks_emits_after_cap(tmp_path: Path) -> None:
    profiler = _build_profiler(
        tmp_path, SpecDecodeProfileMode.METADATA, max_steps=1
    )
    profiler.begin_step(1)
    profiler.emit_worker_batch_metadata(
        SpecDecodeBatchMetadataRecord(
            profile_writer_role="worker",
            profile_writer_id="worker.dp0.pp0.tp0",
            profile_writer_pid=123,
            profile_writer_host="localhost",
            spec_profile_step_id=1,
            speculative_method="ngram",
            batch_size=1,
            req_ids_hash="a",
        )
    )
    transport = profiler.finalize_step_transport()
    assert transport is not None
    assert transport.cost_breakdown is None
    assert profiler._recorded_steps == 1

    profiler.begin_step(2)
    profiler.emit_worker_batch_metadata(
        SpecDecodeBatchMetadataRecord(
            profile_writer_role="worker",
            profile_writer_id="worker.dp0.pp0.tp0",
            profile_writer_pid=123,
            profile_writer_host="localhost",
            spec_profile_step_id=2,
            speculative_method="ngram",
            batch_size=1,
            req_ids_hash="b",
        )
    )
    profiler.close()
    meta_files = list(tmp_path.glob("spec_decode_worker_batch_metadata.*.jsonl"))
    assert len(meta_files) == 1
    assert meta_files[0].read_text(encoding="utf-8").count("\n") == 1


def test_max_steps_blocks_begin_after_cost_finalizes(tmp_path: Path) -> None:
    profiler = _build_profiler(
        tmp_path, SpecDecodeProfileMode.COST_BREAKDOWN, max_steps=1
    )
    profiler.begin_step(1)
    profiler.add_stage_elapsed_ms("target_forward", 1.0)
    assert profiler.finalize_step_transport() is not None
    assert profiler._recorded_steps == 1
    profiler.begin_step(2)
    assert profiler._active_step_id is None
    profiler.add_stage_elapsed_ms("target_forward", 99.0)
    assert profiler.finalize_step_transport() is None
    profiler.close()
    cost_file = next(tmp_path.glob("spec_decode_cost_breakdown.*.jsonl"))
    assert cost_file.read_text(encoding="utf-8").count("\n") == 1


def test_writer_io_failure_is_best_effort(tmp_path: Path, monkeypatch) -> None:
    original_flush_manifest = JsonlProfileWriter._flush_manifest

    def fail_flush_manifest(self) -> None:
        raise OSError("manifest write failure")

    monkeypatch.setattr(JsonlProfileWriter, "_flush_manifest", fail_flush_manifest)
    writer = JsonlProfileWriter(
        output_dir=str(tmp_path),
        profile_writer_role="worker",
        profile_writer_id="worker.dp0.pp0.tp0",
        profile_writer_pid=123,
        profile_writer_host="localhost",
        flush_interval=1,
        mode=SpecDecodeProfileMode.ALL,
        timing_backend=SpecDecodeProfileTimingBackend.DEVICE_SYNC,
        request_sample_rate=1.0,
        max_reqs_per_step_record=None,
        include_token_ids=False,
    )
    assert writer._disabled_due_to_error
    # Should not raise even after failure.
    writer.flush()
    writer.close()
    monkeypatch.setattr(JsonlProfileWriter, "_flush_manifest", original_flush_manifest)


def test_hierarchical_profiler_origin_collapse_helpers() -> None:
    plan = PivotExpansionPlan(
        expanded_to_origin=[0, 0, 1],
        families=[],
        expanded_batch_size=3,
    )
    summed = _collapse_int_sum_by_origin([2, 3, 4], plan, batch_size=2)
    assert summed == [5, 4]
    winner = _collapse_winner_int_by_origin([10, 99, 7], plan, batch_size=2)
    assert winner == [10, 7]
    # Winner-aware partial for origin 0 is the first expanded row (10), not 10+99.
    assert winner[0] != 10 + 99
