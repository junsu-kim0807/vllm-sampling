"""Unified speculative profiling types for vLLM V1.

The profiling join semantics are intentionally explicit:
- ``spec_profile_step_id`` is local monotonic per writer.
- Canonical request-level join key is
  ``profile_writer_id + spec_profile_step_id + req_id``.
- ``req_index`` is auxiliary/debug context only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch


class SpecDecodeProfileMode(str, Enum):
    DISABLED = "disabled"
    COST_BREAKDOWN = "cost_breakdown"
    METADATA = "metadata"
    ALL = "all"

    @property
    def cost_enabled(self) -> bool:
        return self in (self.COST_BREAKDOWN, self.ALL)

    @property
    def metadata_enabled(self) -> bool:
        return self in (self.METADATA, self.ALL)


class SpecDecodeProfileTimingBackend(str, Enum):
    CUDA_EVENT = "cuda_event"
    DEVICE_SYNC = "device_sync"


@dataclass
class PendingCudaEventSpan:
    stage_name: str
    start_event: torch.cuda.Event | None = None
    end_event: torch.cuda.Event | None = None
    start_perf_counter: float | None = None
    elapsed_ms: float | None = None
    completed: bool = False


@dataclass
class SpecDecodeCostBreakdownRecord:
    """Batch-only cost record for one writer-local profiling step.

    Time fields represent per-writer local elapsed stage time aggregated over
    all speculative ubatches in one scheduler step. They do not represent
    end-to-end request latency.

    ``draft_forward_time_ms`` is inclusive of work inside ``propose_draft_token_ids``
    including hierarchical staging. ``partial_verification_time_ms`` and
    ``intermediate_verification_time_ms`` are nested attribution buckets for
    analysis; they are not disjoint wall-clock partitions—do not assume their
    sum with ``draft_forward_time_ms`` equals total draft wall time.

    Per-request token lists are provisional (pre-target) semantics, not final
    scheduler accept/reject. See UNIFIED_PROFILER_NOTE.md for definitions and
    per-metric origin-collapse rules (sum vs winner-aware partial).
    """

    profile_writer_role: str
    profile_writer_id: str
    profile_writer_pid: int
    profile_writer_host: str
    spec_profile_step_id: int
    target_forward_time_ms: float = 0.0
    draft_forward_time_ms: float = 0.0
    reject_sample_time_ms: float = 0.0
    bookkeeping_time_ms: float = 0.0
    partial_verification_time_ms: float = 0.0
    intermediate_verification_time_ms: float = 0.0
    # None: worker has not populated partial acceptance (not staged verification).
    # Do not interpret as real per-request partial counts in analysis or stats.
    num_partial_accepted_per_req: list[int] | None = None
    num_intermediate_accepted_per_req: list[int] | None = None
    num_intermediate_verified_tokens_per_req: list[int] | None = None
    staged_verification_depth: int | None = None
    staged_verification_used: bool | None = None


@dataclass
class SpecDecodeBatchMetadataRecord:
    profile_writer_role: str
    profile_writer_id: str
    profile_writer_pid: int
    profile_writer_host: str
    spec_profile_step_id: int
    speculative_method: str | None
    batch_size: int
    req_ids: list[str] | None = None
    req_ids_hash: str | None = None
    req_id_to_index: dict[str, int] | None = None
    total_num_scheduled_tokens: int = 0
    num_tokens_unpadded: int = 0
    num_tokens_padded: int = 0
    max_query_len: int = 0
    speculative_decode_active: bool = False
    input_fits_in_drafter: bool | None = None
    use_gpu_sampled_tokens_for_drafting: bool | None = None
    cudagraph_mode: str | None = None


@dataclass
class SpecDecodeRequestMetadataRecord:
    """Request-only scheduler truth record for one speculative request."""

    profile_writer_role: str
    profile_writer_id: str
    profile_writer_pid: int
    profile_writer_host: str
    spec_profile_step_id: int
    req_id: str
    req_index: int | None
    num_draft_tokens: int
    num_accepted_tokens: int
    num_rejected_tokens: int
    num_invalid_spec_tokens: int
    num_partial_accepted_tokens: int | None = None
    num_intermediate_verified_tokens: int | None = None
    num_intermediate_accepted_tokens: int | None = None
    staged_verification_depth: int | None = None
    staged_verification_used: bool | None = None
    num_computed_tokens_before: int | None = None
    num_computed_tokens_after: int | None = None
    num_output_placeholders_before: int | None = None
    num_output_placeholders_after: int | None = None
    finish_reason: str | None = None
    stop_reason: str | int | None = None
    output_placeholder_adjustment: int | None = None


@dataclass
class SpecDecodeFamilyMetadataRecord:
    """Optional pivot/family tracing (metadata mode).

    ``family_stage`` (and legacy ``stage``) values such as ``expand`` /
    ``collapse`` are a **provisional** schema tied to batch reshape boundaries;
    a future revision may prefer stage-centric labels (e.g. ``intermediate``,
    ``target``). Consumers must treat unknown stage strings as opaque.
    """

    profile_writer_role: str
    profile_writer_id: str
    profile_writer_pid: int
    profile_writer_host: str
    spec_profile_step_id: int
    family_id: str
    parent_req_id: str | None = None
    stage: str | None = None
    variant_rank: int | None = None
    family_size_before_prune: int | None = None
    family_size_after_prune: int | None = None
    collapse_winner: str | None = None
    rescued_misspeculation: bool | None = None
    # Planned schema extensions (Phase 6 emission); optional for JSONL compatibility.
    origin_req_id: str | None = None
    # Provisional literals; may evolve — see class docstring.
    family_stage: str | None = None
    num_verified_tokens: int | None = None
    num_accepted_tokens: int | None = None
    family_width_before: int | None = None
    family_width_after: int | None = None
    pruned_variant_count: int | None = None


@dataclass
class SpecDecodeProfileTransport:
    """Compact worker->scheduler transport payload."""

    spec_profile_step_id: int
    cost_breakdown: SpecDecodeCostBreakdownRecord | None = None
