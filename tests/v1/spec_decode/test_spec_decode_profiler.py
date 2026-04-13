# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the unified speculative decode profiler.

Unit tests use synthetic/mock data for deterministic verification.
Integration tests (marked) use relaxed bounds.
"""

import json
import os
import tempfile
from dataclasses import asdict
from unittest.mock import MagicMock, patch

import pytest

from vllm.v1.spec_decode.profiler_types import (
    KERNEL_CATEGORY_MAP_V1,
    KERNEL_MAP_VERSION,
    KernelConfidence,
    SpecDecodeBatchMetadataRecord,
    SpecDecodeFamilyMetadataRecord,
    SpecDecodeProfileMode,
    SpecDecodeProfileTimingBackend,
    SpecDecodeProfileTransport,
    SpecDecodeRequestMetadataRecord,
    SpecDecodeStageCostRecord,
    SpecDecodeStageKernelRecord,
    SpecDecodeStageMemoryRecord,
    SpecDecodeStageShapeRecord,
    StepProfileContext,
    _MemorySnapshotState,
)
from vllm.v1.spec_decode.profiler import (
    DeepSpecDecodeKernelProfiler,
    JsonlProfileWriter,
    NullSpecDecodeProfiler,
    OnlineSpecDecodeProfiler,
)
from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID
from vllm.v1.spec_decode.spec_stage_ops import (
    select_pivot_expanded_rows_to_origin,
)
from vllm.v1.spec_decode.spec_stage_runtime import (
    PivotExpansionFamily,
    PivotExpansionPlan,
)


# ---------------------------------------------------------------------------
# Mode enum tests
# ---------------------------------------------------------------------------


class TestSpecDecodeProfileMode:
    def test_disabled_properties(self):
        m = SpecDecodeProfileMode.DISABLED
        assert not m.cost_enabled
        assert not m.shape_memory_enabled
        assert not m.kernel_enabled
        assert not m.metadata_enabled

    def test_stage_cost_properties(self):
        m = SpecDecodeProfileMode.STAGE_COST
        assert m.cost_enabled
        assert not m.shape_memory_enabled
        assert not m.kernel_enabled
        assert m.metadata_enabled

    def test_shape_memory_properties(self):
        m = SpecDecodeProfileMode.SHAPE_MEMORY
        assert m.cost_enabled
        assert m.shape_memory_enabled
        assert not m.kernel_enabled
        assert m.metadata_enabled

    def test_kernel_breakdown_properties(self):
        m = SpecDecodeProfileMode.KERNEL_BREAKDOWN
        assert m.cost_enabled
        assert not m.shape_memory_enabled
        assert m.kernel_enabled
        assert m.metadata_enabled

    def test_all_properties(self):
        m = SpecDecodeProfileMode.ALL
        assert m.cost_enabled
        assert m.shape_memory_enabled
        assert m.kernel_enabled
        assert m.metadata_enabled


# ---------------------------------------------------------------------------
# NullProfiler tests
# ---------------------------------------------------------------------------


class TestNullProfiler:
    def test_disabled_no_op(self):
        p = NullSpecDecodeProfiler()
        ctx = p.begin_step(42, "spec_verify")
        assert ctx.step_id == 42
        assert ctx.forward_reason == "spec_verify"
        p.start_stage("draft_forward")
        elapsed = p.end_stage("draft_forward")
        assert elapsed == 0.0
        transport = p.finalize_step(ctx)
        assert transport is None

    def test_snapshot_methods_no_error(self):
        p = NullSpecDecodeProfiler()
        p.snapshot_shape("x", {"origin_batch_size": 1})
        p.snapshot_memory_before("x")
        p.snapshot_memory_after("x")
        p.record_kv_proxies("x")
        p.emit_batch_metadata(SpecDecodeBatchMetadataRecord())
        p.emit_request_metadata(SpecDecodeRequestMetadataRecord())
        p.emit_family_metadata(SpecDecodeFamilyMetadataRecord())
        p.close()


# ---------------------------------------------------------------------------
# StepProfileContext lifecycle tests
# ---------------------------------------------------------------------------


class TestStepProfileContext:
    def test_fresh_context_is_empty(self):
        ctx = StepProfileContext(step_id=1, forward_reason="spec_verify")
        assert ctx.stage_timings == {}
        assert ctx.stage_shapes == []
        assert ctx.stage_memories == []
        assert ctx.kernel_records == []
        assert ctx._pending_memory_snapshots == {}

    def test_bookkeeping_sub_timings_default_zero(self):
        ctx = StepProfileContext()
        assert ctx.bookkeeping_metadata_build_ms == 0.0
        assert ctx.bookkeeping_scheduler_bridge_ms == 0.0
        assert ctx.bookkeeping_output_pack_ms == 0.0

    def test_bookkeeping_sub_timings_accumulate(self):
        ctx = StepProfileContext()
        ctx.bookkeeping_metadata_build_ms += 1.5
        ctx.bookkeeping_scheduler_bridge_ms += 0.3
        ctx.bookkeeping_output_pack_ms += 0.2
        assert ctx.bookkeeping_metadata_build_ms == 1.5
        assert ctx.bookkeeping_scheduler_bridge_ms == 0.3
        assert ctx.bookkeeping_output_pack_ms == 0.2

    def test_no_state_leakage_between_contexts(self):
        c1 = StepProfileContext(step_id=1)
        c1.stage_timings["draft_forward"] = [1.0, 2.0]
        c2 = StepProfileContext(step_id=2)
        assert "draft_forward" not in c2.stage_timings
        assert c2.stage_timings == {}


# ---------------------------------------------------------------------------
# JsonlProfileWriter tests
# ---------------------------------------------------------------------------


class TestJsonlProfileWriter:
    def test_write_and_flush(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = JsonlProfileWriter(tmpdir, "test_writer", flush_interval=2)
            rec1 = SpecDecodeStageCostRecord(
                step_id=1, stage_name="draft_forward", wall_time_ms=1.5)
            rec2 = SpecDecodeStageCostRecord(
                step_id=2, stage_name="target_verify", wall_time_ms=2.5)
            writer.append("stage_cost", rec1)
            writer.append("stage_cost", rec2)
            writer.flush_all()

            path = os.path.join(
                tmpdir, "spec_decode_stage_cost.test_writer.jsonl")
            assert os.path.exists(path)
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 2
            data = json.loads(lines[0])
            assert data["step_id"] == 1
            assert data["stage_name"] == "draft_forward"
            assert data["wall_time_ms"] == 1.5

    def test_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = JsonlProfileWriter(tmpdir, "test_w", flush_interval=1)
            writer.append("stage_cost", SpecDecodeStageCostRecord(step_id=1))
            writer.close()

            manifest_path = os.path.join(
                tmpdir, "spec_decode_profile_manifest.test_w.json")
            assert os.path.exists(manifest_path)
            with open(manifest_path) as f:
                manifest = json.load(f)
            assert manifest["writer_id"] == "test_w"
            assert "stage_cost" in manifest["files"]

    def test_max_steps_cap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = JsonlProfileWriter(tmpdir, "cap", flush_interval=1)
            profiler = OnlineSpecDecodeProfiler(
                mode=SpecDecodeProfileMode.STAGE_COST,
                timing_backend=SpecDecodeProfileTimingBackend.DEVICE_SYNC,
                writer=writer,
                writer_id="cap",
                max_steps=2,
            )
            for i in range(5):
                ctx = profiler.begin_step(i, "decode")
                ctx.stage_timings["draft_forward"] = [1.0]
                transport = profiler.finalize_step(ctx)
                if i < 2:
                    assert transport is not None
                else:
                    assert transport is None


# ---------------------------------------------------------------------------
# OnlineSpecDecodeProfiler tests
# ---------------------------------------------------------------------------


class TestOnlineProfiler:
    def test_stage_cost_emits_transport(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = JsonlProfileWriter(tmpdir, "w", flush_interval=1)
            profiler = OnlineSpecDecodeProfiler(
                mode=SpecDecodeProfileMode.STAGE_COST,
                timing_backend=SpecDecodeProfileTimingBackend.DEVICE_SYNC,
                writer=writer,
                writer_id="w",
            )
            ctx = profiler.begin_step(10, "spec_verify")
            ctx.stage_timings["target_verify"] = [5.0]
            transport = profiler.finalize_step(ctx)
            assert transport is not None
            assert transport.step_id == 10
            assert transport.cost_record is not None
            assert transport.cost_record.forward_reason == "spec_verify"
            assert "target_verify" in transport.stage_times_ms

    def test_forward_reason_correctly_tagged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = JsonlProfileWriter(tmpdir, "w", flush_interval=1)
            profiler = OnlineSpecDecodeProfiler(
                mode=SpecDecodeProfileMode.STAGE_COST,
                timing_backend=SpecDecodeProfileTimingBackend.DEVICE_SYNC,
                writer=writer,
                writer_id="w",
            )
            for reason in ("spec_verify", "prefill", "decode"):
                ctx = profiler.begin_step(0, reason)
                ctx.stage_timings["x"] = [1.0]
                transport = profiler.finalize_step(ctx)
                assert transport.cost_record.forward_reason == reason

    def test_begin_step_creates_fresh_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = JsonlProfileWriter(tmpdir, "w", flush_interval=1)
            profiler = OnlineSpecDecodeProfiler(
                mode=SpecDecodeProfileMode.STAGE_COST,
                timing_backend=SpecDecodeProfileTimingBackend.DEVICE_SYNC,
                writer=writer,
                writer_id="w",
            )
            ctx1 = profiler.begin_step(1, "decode")
            ctx1.stage_timings["draft_forward"] = [10.0]
            profiler.finalize_step(ctx1)

            ctx2 = profiler.begin_step(2, "decode")
            assert ctx2.step_id == 2
            assert ctx2.stage_timings == {}
            assert profiler._ctx is ctx2

    def test_finalize_clears_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = JsonlProfileWriter(tmpdir, "w", flush_interval=1)
            profiler = OnlineSpecDecodeProfiler(
                mode=SpecDecodeProfileMode.STAGE_COST,
                timing_backend=SpecDecodeProfileTimingBackend.DEVICE_SYNC,
                writer=writer,
                writer_id="w",
            )
            ctx = profiler.begin_step(1, "decode")
            profiler.finalize_step(ctx)
            assert profiler._ctx is None


# ---------------------------------------------------------------------------
# DeepSpecDecodeKernelProfiler tests
# ---------------------------------------------------------------------------


class TestDeepKernelProfiler:
    def test_sample_gating_deterministic(self):
        p = DeepSpecDecodeKernelProfiler(
            writer_id="w", sample_rate=1.0)
        assert p.should_profile_step(0)
        assert p.should_profile_step(999)

    def test_sample_gating_zero_rate(self):
        p = DeepSpecDecodeKernelProfiler(
            writer_id="w", sample_rate=0.0)
        assert not p.should_profile_step(0)

    def test_stage_filter(self):
        p = DeepSpecDecodeKernelProfiler(
            writer_id="w", sample_rate=1.0,
            stage_filter=["target_verify"])
        assert p.should_profile_stage("target_verify")
        assert not p.should_profile_stage("draft_forward")
        assert not p.should_profile_stage("intermediate_verify")

    def test_stage_filter_none_allows_all(self):
        p = DeepSpecDecodeKernelProfiler(
            writer_id="w", sample_rate=1.0,
            stage_filter=None)
        assert p.should_profile_stage("target_verify")
        assert p.should_profile_stage("draft_forward")

    def test_classify_attention_kernel_direct(self):
        p = DeepSpecDecodeKernelProfiler(writer_id="w")
        event = MagicMock()
        event.name = "flash_attn_fwd_kernel"
        cat, conf = p._classify_single_kernel("flash_attn_fwd_kernel", event, 1.0)
        assert cat == "kv_sensitive_attention"
        assert conf == KernelConfidence.DIRECT.value

    def test_classify_nccl_direct(self):
        p = DeepSpecDecodeKernelProfiler(writer_id="w")
        event = MagicMock()
        event.name = "nccl_allreduce_kernel"
        cat, conf = p._classify_single_kernel("nccl_allreduce_kernel", event, 1.0)
        assert cat == "communication"
        assert conf == KernelConfidence.DIRECT.value

    def test_classify_sampling_direct(self):
        p = DeepSpecDecodeKernelProfiler(writer_id="w")
        event = MagicMock()
        event.name = "top_k_sampling"
        cat, conf = p._classify_single_kernel("top_k_sampling", event, 1.0)
        assert cat == "sampling"
        assert conf == KernelConfidence.DIRECT.value

    def test_classify_gemm_memory_dominated_by_ai(self):
        p = DeepSpecDecodeKernelProfiler(
            writer_id="w", gemm_ai_threshold=50.0)
        event = MagicMock()
        event.name = "cublas_gemm_small"
        event.stack = []
        # M=1, K=4096, N=4096 => FLOPs=2*1*4096*4096=33.5M, bytes=(1*4096+4096*4096+1*4096)*2=33.6M
        # AI = 33.5M / 33.6M ≈ 1.0 (memory dominated)
        event.input_shapes = [[1, 4096], [4096, 4096]]
        cat, conf = p._classify_single_kernel("cublas_gemm_small", event, 1.0)
        assert cat == "gemm_memory_dominated"
        assert conf == KernelConfidence.PROXY.value

    def test_classify_gemm_compute_dominated_by_ai(self):
        p = DeepSpecDecodeKernelProfiler(
            writer_id="w", gemm_ai_threshold=50.0)
        event = MagicMock()
        event.name = "cublas_gemm_large"
        event.stack = []
        # M=1024, K=4096, N=4096 => FLOPs=2*1024*4096*4096=34.4G, bytes=(1024*4096+4096*4096+1024*4096)*2=50.3M
        # AI = 34.4G / 50.3M ≈ 683 (compute dominated)
        event.input_shapes = [[1024, 4096], [4096, 4096]]
        cat, conf = p._classify_single_kernel("cublas_gemm_large", event, 1.0)
        assert cat == "gemm_compute_dominated"
        assert conf == KernelConfidence.PROXY.value

    def test_classify_gemm_with_mlp_stack_context(self):
        p = DeepSpecDecodeKernelProfiler(
            writer_id="w", gemm_ai_threshold=50.0)
        event = MagicMock()
        event.name = "cutlass_gemm"
        event.stack = ["model.layers.0.mlp.gate_up_proj", "forward"]
        # Large GEMM with MLP context -> compute dominated via derived
        event.input_shapes = [[1024, 4096], [4096, 8192]]
        cat, conf = p._classify_single_kernel("cutlass_gemm", event, 1.0)
        assert cat == "gemm_compute_dominated"
        assert conf == KernelConfidence.DERIVED.value

    def test_classify_gemm_with_attention_stack_context(self):
        p = DeepSpecDecodeKernelProfiler(
            writer_id="w", gemm_ai_threshold=50.0)
        event = MagicMock()
        event.name = "cutlass_gemm"
        event.stack = ["model.layers.0.self_attn.q_proj", "forward"]
        event.input_shapes = [[1024, 4096], [4096, 4096]]
        cat, conf = p._classify_single_kernel("cutlass_gemm", event, 1.0)
        assert cat == "kv_sensitive_attention"
        assert conf == KernelConfidence.DERIVED.value

    def test_unattributed_time_exact_for_mocks(self):
        """Synthetic mock events: unattributed = wall - sum(categorized)."""
        p = DeepSpecDecodeKernelProfiler(writer_id="w")

        mock_events = []
        # Attention event
        ev1 = MagicMock()
        ev1.name = "flash_attn_fwd"
        ev1.self_cuda_time_total = 5000  # 5ms in µs
        ev1.self_cpu_time_total = 0
        ev1.input_shapes = []
        ev1.stack = []
        mock_events.append(ev1)
        # NCCL event
        ev2 = MagicMock()
        ev2.name = "nccl_all_reduce"
        ev2.self_cuda_time_total = 2000  # 2ms
        ev2.self_cpu_time_total = 0
        ev2.input_shapes = []
        ev2.stack = []
        mock_events.append(ev2)

        mock_prof = MagicMock()
        mock_prof.events.return_value = mock_events

        ctx = StepProfileContext(step_id=1, forward_reason="spec_verify")
        record = p._classify_events(
            mock_prof, "target_verify", 0,
            wall_time_ms=10.0, ctx=ctx,
            origin_batch_size=4, effective_batch_size=4,
            num_tokens=100, sum_seq_lens=400, max_seq_len=128)

        assert record.kv_sensitive_attention_time_ms == pytest.approx(5.0)
        assert record.communication_time_ms == pytest.approx(2.0)
        expected_unattr = 10.0 - 5.0 - 2.0
        assert record.unattributed_time_ms == pytest.approx(
            expected_unattr, abs=0.01)

    def test_confidence_tags_assigned_correctly(self):
        p = DeepSpecDecodeKernelProfiler(writer_id="w")

        ev1 = MagicMock()
        ev1.name = "flash_attn_fwd"
        ev1.self_cuda_time_total = 1000
        ev1.self_cpu_time_total = 0
        ev1.input_shapes = []
        ev1.stack = []

        mock_prof = MagicMock()
        mock_prof.events.return_value = [ev1]

        ctx = StepProfileContext(step_id=1, forward_reason="spec_verify")
        record = p._classify_events(
            mock_prof, "target_verify", 0,
            wall_time_ms=5.0, ctx=ctx,
            origin_batch_size=1, effective_batch_size=1,
            num_tokens=10, sum_seq_lens=10, max_seq_len=10)

        assert record.kv_sensitive_attention_confidence == KernelConfidence.DIRECT.value


# ---------------------------------------------------------------------------
# Record serialization tests
# ---------------------------------------------------------------------------


class TestRecordSerialization:
    def test_cost_record_to_dict(self):
        rec = SpecDecodeStageCostRecord(
            step_id=1, stage_name="draft_forward",
            wall_time_ms=1.5, forward_reason="spec_verify")
        d = asdict(rec)
        assert d["step_id"] == 1
        assert d["wall_time_ms"] == 1.5

    def test_shape_record_kv_working_set(self):
        rec = SpecDecodeStageShapeRecord(
            kv_working_set_pages=42,
            kv_live_pages_total=100)
        d = asdict(rec)
        assert d["kv_working_set_pages"] == 42
        assert d["kv_live_pages_total"] == 100

    def test_memory_record_reserved_peak(self):
        rec = SpecDecodeStageMemoryRecord(
            alloc_peak_bytes=1000,
            reserved_peak_bytes=2000)
        d = asdict(rec)
        assert d["alloc_peak_bytes"] == 1000
        assert d["reserved_peak_bytes"] == 2000

    def test_kernel_record_gemm_split(self):
        rec = SpecDecodeStageKernelRecord(
            gemm_memory_dominated_time_ms=3.0,
            gemm_memory_confidence=KernelConfidence.PROXY.value,
            gemm_compute_dominated_time_ms=7.0,
            gemm_compute_confidence=KernelConfidence.DERIVED.value)
        d = asdict(rec)
        assert d["gemm_memory_dominated_time_ms"] == 3.0
        assert d["gemm_compute_confidence"] == "derived"

    def test_family_metadata_expanded_fields(self):
        rec = SpecDecodeFamilyMetadataRecord(
            winner_family_row_idx=2,
            winner_accept_len=3,
            topk_k=5,
            expansion_pct=0.4,
            collapse_reason="winner_selected",
            selected_expansion_req_ids=["r1", "r2"])
        d = asdict(rec)
        assert d["winner_accept_len"] == 3
        assert d["collapse_reason"] == "winner_selected"

    def test_transport_stage_times(self):
        t = SpecDecodeProfileTransport(
            step_id=1,
            stage_times_ms={"draft_forward": 3.0, "target_verify": 5.0})
        assert t.stage_times_ms["draft_forward"] == 3.0


# ---------------------------------------------------------------------------
# Origin collapse helpers
# ---------------------------------------------------------------------------


class TestOriginCollapseHelpers:
    def test_sum_collapse(self):
        """Sum collapse: total verified tokens across expanded rows."""
        per_row = [3, 2, 4, 1]
        assert sum(per_row) == 10

    def test_winner_aware_collapse(self):
        """Winner-aware: pick first expanded row per origin (simulated)."""
        origin_to_rows = {0: [0, 1], 1: [2, 3]}
        per_row_partial = [5, 3, 4, 2]
        winner_partial = [per_row_partial[rows[0]]
                          for rows in origin_to_rows.values()]
        assert winner_partial == [5, 4]

    def test_select_pivot_expanded_rows_prefers_longer_accept(self):
        plan = PivotExpansionPlan(
            expanded_to_origin=[0, 1, 0, 1],
            families=[
                PivotExpansionFamily(
                    origin_row=0,
                    expanded_rows=[0, 2],
                    candidate_ranks=[0, 1],
                    first_token_ids=[11, 12],
                    first_token_probs=[0.9, 0.4],
                ),
                PivotExpansionFamily(
                    origin_row=1,
                    expanded_rows=[1, 3],
                    candidate_ranks=[0, 1],
                    first_token_ids=[21, 22],
                    first_token_probs=[0.8, 0.3],
                ),
            ],
            expanded_batch_size=4,
            origin_batch_size=2,
            packed_row_is_active=[True, True, True, True],
            origin_to_base_row=[0, 1],
        )
        sampled = pytest.importorskip("torch").tensor([
            [1, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID],
            [2, 3, PLACEHOLDER_TOKEN_ID],
            [4, 5, 6],
            [7, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID],
        ])
        selected = select_pivot_expanded_rows_to_origin(
            sampled,
            plan,
            num_draft_tokens=[3, 3, 3, 3],
        )
        assert selected == [2, 1]


# ---------------------------------------------------------------------------
# Kernel category map sanity tests
# ---------------------------------------------------------------------------


class TestKernelCategoryMap:
    def test_map_version(self):
        assert KERNEL_MAP_VERSION == "v1"

    def test_attention_patterns_present(self):
        assert "flash_attn" in KERNEL_CATEGORY_MAP_V1["attention_direct"]
        assert "paged_attention" in KERNEL_CATEGORY_MAP_V1["attention_direct"]

    def test_communication_patterns_present(self):
        assert "nccl" in KERNEL_CATEGORY_MAP_V1["communication_direct"]
        assert "all_reduce" in KERNEL_CATEGORY_MAP_V1["communication_direct"]


# ---------------------------------------------------------------------------
# SpecDecodingStats stage time fold tests (scheduler-side)
# ---------------------------------------------------------------------------


class TestSchedulerFold:
    """Test that transport stage_times_ms fold into SpecDecodingStats."""

    def test_fold_maps_all_five_stages(self):
        from vllm.v1.spec_decode.metrics import SpecDecodingStats

        stats = SpecDecodingStats.new(num_spec_tokens=5)
        transport = SpecDecodeProfileTransport(
            step_id=1,
            stage_times_ms={
                "draft_forward": 3.0,
                "target_verify": 5.0,
                "intermediate_verify": 2.0,
                "expand_collapse": 1.0,
                "reject_sample": 500.0,  # ms
            },
        )
        # Simulate what _fold_batch_cost_from_transport does.
        stage_ms = transport.stage_times_ms
        stats.draft_forward_time_ms += stage_ms.get("draft_forward", 0.0)
        stats.target_verify_time_ms += stage_ms.get("target_verify", 0.0)
        stats.intermediate_verify_time_ms += stage_ms.get(
            "intermediate_verify", 0.0)
        stats.expand_collapse_time_ms += stage_ms.get(
            "expand_collapse", 0.0)
        stats.reject_sample_time_sec += stage_ms.get(
            "reject_sample", 0.0) / 1000.0

        assert stats.draft_forward_time_ms == 3.0
        assert stats.target_verify_time_ms == 5.0
        assert stats.intermediate_verify_time_ms == 2.0
        assert stats.expand_collapse_time_ms == 1.0
        assert stats.reject_sample_time_sec == pytest.approx(0.5)

    def test_step_id_monotonic(self):
        """Verify that step IDs in transport are monotonically increasing."""
        transports = [
            SpecDecodeProfileTransport(step_id=i)
            for i in range(10)
        ]
        ids = [t.step_id for t in transports]
        assert ids == list(range(10))

    def test_fold_with_no_transport(self):
        from vllm.v1.spec_decode.metrics import SpecDecodingStats

        stats = SpecDecodingStats.new(num_spec_tokens=5)
        stats.draft_forward_time_ms = 1.0
        # No transport — should not change stats.
        assert stats.draft_forward_time_ms == 1.0


# ---------------------------------------------------------------------------
# First-draft top-k metadata helpers
# ---------------------------------------------------------------------------


class TestFirstDraftTopkMetadataHelpers:
    def test_req_id_row_index_map_first_wins(self):
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        m = GPUModelRunner._req_id_row_index_map(("r0", "r1", "r0"))
        assert m == {"r0": 0, "r1": 1}

    def test_first_draft_topk_for_profile_row(self):
        import torch

        from vllm.v1.spec_decode.spec_stage_runtime import RootTopKInfo
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        tid = torch.tensor([[10, 20, 30], [40, 50, 60]], dtype=torch.long)
        pr = torch.tensor(
            [[0.5, 0.3, 0.2], [0.7, 0.2, 0.1]], dtype=torch.float32
        )
        info = RootTopKInfo(topk_token_ids=tid, topk_probs=pr)
        ids, conf, k = GPUModelRunner._first_draft_topk_for_profile_row(
            1, info, k=2)
        assert ids == [40, 50]
        assert conf == pytest.approx([0.7, 0.2])
        assert k == 2

    def test_request_metadata_record_first_draft_fields_json(self):
        rec = SpecDecodeRequestMetadataRecord(
            req_id="a",
            first_draft_topk_token_ids=[1, 2],
            first_draft_topk_confidences=[0.9, 0.1],
            first_draft_topk_k=2,
            first_draft_topk_source="profile_first_draft_topk",
        )
        d = asdict(rec)
        assert d["first_draft_topk_k"] == 2
        assert d["first_draft_topk_source"] == "profile_first_draft_topk"
