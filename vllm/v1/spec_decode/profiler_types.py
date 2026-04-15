# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Type definitions for the unified speculative decode profiler.

Two-layer profiler:
  Layer A (Online) — always-on, stage wall-time + shape/memory + KV proxies
  Layer B (Deep)   — sampled steps, raw CUDA event kernel classification
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SpecDecodeProfileMode(str, enum.Enum):
    """Profiling verbosity."""
    DISABLED = "disabled"
    STAGE_COST = "stage_cost"
    SHAPE_MEMORY = "shape_memory"
    KERNEL_BREAKDOWN = "kernel_breakdown"
    ALL = "all"

    @property
    def cost_enabled(self) -> bool:
        return self is not SpecDecodeProfileMode.DISABLED

    @property
    def shape_memory_enabled(self) -> bool:
        return self in (SpecDecodeProfileMode.SHAPE_MEMORY,
                        SpecDecodeProfileMode.ALL)

    @property
    def kernel_enabled(self) -> bool:
        return self in (SpecDecodeProfileMode.KERNEL_BREAKDOWN,
                        SpecDecodeProfileMode.ALL)

    @property
    def metadata_enabled(self) -> bool:
        return self is not SpecDecodeProfileMode.DISABLED


class SpecDecodeProfileTimingBackend(str, enum.Enum):
    CUDA_EVENT = "cuda_event"
    DEVICE_SYNC = "device_sync"


class KernelConfidence(str, enum.Enum):
    """How much to trust a kernel-category assignment."""
    DIRECT = "direct"
    DERIVED = "derived"
    PROXY = "proxy"


# ---------------------------------------------------------------------------
# Kernel category map (versioned)
# ---------------------------------------------------------------------------

KERNEL_CATEGORY_MAP_V1: dict[str, list[str]] = {
    "attention_direct": [
        "flash_attn", "paged_attention", "flashinfer",
        "sdpa", "attn_fwd", "fmha", "cutlass_fmha",
    ],
    "communication_direct": [
        "nccl", "all_reduce", "all_gather",
        "reduce_scatter", "broadcast",
    ],
    "sampling_direct": [
        "top_k", "top_p", "sample", "multinomial",
        "rejection_sample",
    ],
    "normalization_direct": [
        "layernorm", "rmsnorm", "rms_norm",
    ],
    "activation_direct": [
        "silu_mul", "gelu", "relu", "swiglu",
    ],
}

GEMM_KERNEL_PATTERNS: list[str] = [
    "gemm", "cutlass", "cublas", "triton_matmul",
]

KERNEL_MAP_VERSION = "v1"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@dataclass
class _MemorySnapshotState:
    """In-flight memory snapshot (before -> waiting for after)."""
    alloc_before: int = 0
    reserved_before: int = 0


@dataclass
class PendingCudaEventSpan:
    """Tracks an in-flight CUDA event timing span."""
    stage_name: str
    invocation_idx: int = 0
    start_event: Any = None  # torch.cuda.Event
    end_event: Any = None    # torch.cuda.Event
    start_perf_counter: float = 0.0
    elapsed_ms: float = 0.0
    completed: bool = False


# ---------------------------------------------------------------------------
# Record types — written to JSONL
# ---------------------------------------------------------------------------

@dataclass
class SpecDecodeStageCostRecord:
    """Per-stage wall-time (one record per stage invocation per step)."""
    profile_writer_id: str = ""
    step_id: int = 0
    stage_name: str = ""
    stage_invocation_idx: int = 0
    forward_reason: str = ""
    wall_time_ms: float = 0.0

    num_intermediate_verified_tokens_per_req: list[int] | None = None
    num_intermediate_accepted_tokens_per_req: list[int] | None = None
    num_partial_accepted_per_req: list[int] | None = None

    staged_verification_depth: int = 0
    staged_verification_used: bool = False

    # Bookkeeping sub-timings (supplementary, not primary stages)
    bookkeeping_metadata_build_ms: float = 0.0
    bookkeeping_scheduler_bridge_ms: float = 0.0
    bookkeeping_output_pack_ms: float = 0.0


@dataclass
class SpecDecodeStageShapeRecord:
    """Per-stage shape snapshot for expansion analysis."""
    profile_writer_id: str = ""
    step_id: int = 0
    stage_name: str = ""
    stage_invocation_idx: int = 0
    forward_reason: str = ""

    origin_batch_size: int = 0
    effective_batch_size: int = 0
    num_expanded_rows: int = 0
    num_families: int = 0
    avg_family_size: float = 0.0
    max_family_size: int = 0
    num_tokens_processed: int = 0
    sum_seq_lens: int = 0
    max_seq_len: int = 0

    # KV proxies
    kv_load_work_proxy: int = 0
    kv_working_set_pages: int = 0
    kv_live_pages_total: int = 0
    num_active_kv_pages_estimate: int = 0

    # Expansion config
    uses_topk_expansion: bool = False
    topk_k: int | None = None
    expansion_pct: float = 0.0
    uses_intermediate_verification: bool = False
    intermediate_rounds: int | None = None
    verification_chunk_len: int | None = None


@dataclass
class SpecDecodeStageMemoryRecord:
    """Per-stage memory inflation snapshot."""
    profile_writer_id: str = ""
    step_id: int = 0
    stage_name: str = ""
    stage_invocation_idx: int = 0

    alloc_before_bytes: int = 0
    alloc_after_bytes: int = 0
    alloc_peak_bytes: int = 0
    reserved_before_bytes: int = 0
    reserved_after_bytes: int = 0
    reserved_peak_bytes: int = 0

    kv_live_bytes_estimate: int = 0
    kv_working_set_bytes_estimate: int = 0


@dataclass
class SpecDecodeStageKernelRecord:
    """Deep profiler approximate kernel time decomposition (sampled steps)."""
    profile_writer_id: str = ""
    step_id: int = 0
    stage_name: str = ""
    stage_invocation_idx: int = 0
    forward_reason: str = ""

    origin_batch_size: int = 0
    effective_batch_size: int = 0
    num_tokens: int = 0
    sum_seq_lens: int = 0
    max_seq_len: int = 0

    # Time fields (all approximate)
    wall_time_ms: float = 0.0
    kv_sensitive_attention_time_ms: float = 0.0
    kv_sensitive_attention_confidence: str = KernelConfidence.PROXY.value
    attention_other_time_ms: float = 0.0
    gemm_memory_dominated_time_ms: float = 0.0
    gemm_memory_confidence: str = KernelConfidence.PROXY.value
    gemm_compute_dominated_time_ms: float = 0.0
    gemm_compute_confidence: str = KernelConfidence.PROXY.value
    other_compute_time_ms: float = 0.0
    communication_time_ms: float = 0.0
    communication_confidence: str = KernelConfidence.PROXY.value
    sampling_time_ms: float = 0.0
    host_overhead_time_ms: float = 0.0
    unattributed_time_ms: float = 0.0

    kernel_map_version: str = KERNEL_MAP_VERSION


@dataclass
class SpecDecodeRequestMetadataRecord:
    """Scheduler truth per request."""
    profile_writer_id: str = ""
    step_id: int = 0
    req_id: str = ""
    req_index: int = 0

    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0
    num_rejected_tokens: int = 0
    num_invalid_spec_tokens: int = 0

    num_partial_accepted_tokens: int = 0
    num_intermediate_verified_tokens: int = 0
    num_intermediate_accepted_tokens: int = 0

    staged_verification_depth: int = 0
    staged_verification_used: bool = False

    num_computed_tokens_before: int = 0
    num_computed_tokens_after: int = 0
    num_output_placeholders_before: int = 0
    num_output_placeholders_after: int = 0
    finish_reason: str | None = None
    stop_reason: str | None = None

    # Optional raw token arrays (only populated when include_token_ids=True)
    draft_token_ids: list[int] | None = None
    accepted_token_ids: list[int] | None = None

    # First speculative position top-k (profile cache; req_id-aligned via snapshot)
    first_draft_topk_token_ids: list[int] | None = None
    first_draft_topk_confidences: list[float] | None = None
    first_draft_topk_k: int = 0
    first_draft_topk_source: str | None = None

    # Target model at first draft verify position (same pipeline as rejection verify)
    first_draft_target_top1_token_id: int | None = None
    first_draft_target_top1_confidence: float | None = None
    first_draft_target_top1_source: str | None = None


@dataclass
class SpecDecodeBatchMetadataRecord:
    """Worker batch execution context (lightweight, every step)."""
    profile_writer_id: str = ""
    step_id: int = 0

    speculative_method: str = ""
    batch_size: int = 0
    req_ids_hash: str = ""
    total_num_scheduled_tokens: int = 0
    num_tokens_unpadded: int = 0
    num_tokens_padded: int = 0
    max_query_len: int = 0
    speculative_decode_active: bool = False
    input_fits_in_drafter: bool = False
    cudagraph_mode: str = ""


@dataclass
class SpecDecodeFamilyMetadataRecord:
    """Family scaffold for pivot expansion analysis."""
    profile_writer_id: str = ""
    step_id: int = 0
    family_id: str = ""
    origin_req_id: str = ""
    family_stage: str = ""  # "expand" / "collapse"

    family_width_before: int = 0
    family_width_after: int = 0
    pruned_variant_count: int = 0

    winner_family_row_idx: int = -1
    winner_accept_len: int = 0
    topk_k: int | None = None
    expansion_pct: float = 0.0
    collapse_reason: str = ""  # "winner_selected"/"all_rejected"/"timeout"
    selected_expansion_req_ids: list[str] | None = None


# ---------------------------------------------------------------------------
# Transport: compact worker-to-scheduler payload
# ---------------------------------------------------------------------------

@dataclass
class SpecDecodeProfileTransport:
    """Lightweight payload sent from worker to scheduler via ModelRunnerOutput."""
    step_id: int = 0
    cost_record: SpecDecodeStageCostRecord | None = None
    # Per-stage wall-time totals (ms) for scheduler-side Prometheus fold.
    stage_times_ms: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# StepProfileContext — per-step mutable accumulator
# ---------------------------------------------------------------------------

@dataclass
class StepProfileContext:
    """Per-step mutable context created at begin_step, consumed at finalize_step.

    Replaces the old _hv_profiler_* ephemeral stash approach. The context is
    step-scoped and never leaks state across steps.
    """
    step_id: int = 0
    forward_reason: str = ""  # "spec_verify" / "prefill" / "decode"

    # Per-stage accumulators (populated by online profiler)
    stage_timings: dict[str, list[float]] = field(default_factory=dict)
    stage_shapes: list[SpecDecodeStageShapeRecord] = field(
        default_factory=list)
    stage_memories: list[SpecDecodeStageMemoryRecord] = field(
        default_factory=list)

    # Per-request token arrays (accumulated during HV rounds)
    inter_verified_per_req: list[int] | None = None
    inter_accepted_per_req: list[int] | None = None
    partial_accepted_per_req: list[int] | None = None

    # Memory snapshot state (in-flight, for before/after pairing)
    _pending_memory_snapshots: dict[str, _MemorySnapshotState] = field(
        default_factory=dict)

    # Bookkeeping sub-timings (derived, not primary stages)
    bookkeeping_metadata_build_ms: float = 0.0
    bookkeeping_scheduler_bridge_ms: float = 0.0
    bookkeeping_output_pack_ms: float = 0.0

    # Staged verification metadata
    staged_verification_depth: int = 0
    staged_verification_used: bool = False

    # req_id -> (token_id, softmax confidence) for first draft row (verify logits)
    first_draft_target_top1_by_req_id: dict[str, tuple[int, float]] = field(
        default_factory=dict)

    # Deep profiler kernel records (populated only on sampled steps)
    kernel_records: list[SpecDecodeStageKernelRecord] = field(
        default_factory=list)
