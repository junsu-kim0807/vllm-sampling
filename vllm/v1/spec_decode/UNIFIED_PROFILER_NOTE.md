# Unified Speculative Profiler Developer Note

## Which file answers which question
- `spec_decode_cost_breakdown.{profile_writer_id}.jsonl`
  - Per-writer, per-step speculative stage cost (batch-only timing).
- `spec_decode_worker_batch_metadata.{profile_writer_id}.jsonl`
  - Worker batch shape/execution context for interpreting cost regimes.
- `spec_decode_scheduler_request_metadata.{profile_writer_id}.jsonl`
  - Scheduler truth for accepted/rejected/invalid speculative outcomes.
- `spec_decode_family_metadata.{profile_writer_id}.jsonl`
  - Family-aware scaffold for pivot/hierarchical experiments (currently optional).
- `spec_decode_profile_manifest.{profile_writer_id}.json`
  - Run metadata, schema/versioning, writer identity, and emitted file index.

## Join keys and contracts
- Canonical request-level join key:
  - `profile_writer_id + spec_profile_step_id + req_id`
- `spec_profile_step_id` is local monotonic per writer.
- `req_index` is auxiliary/debug context only.
- Cost timing records are batch-only and must not be duplicated per request.

## Worker batch token fields
- `total_num_scheduled_tokens`: scheduler-reported sum of scheduled tokens this step.
- `num_tokens_unpadded` / `num_tokens_padded`: values from the target forward batch
  descriptor in `GPUModelRunner` (padding/CUDA graph path can make padded larger
  than unpadded). Do not substitute `total_num_scheduled_tokens` for both.

## Partial acceptance in cost records
- `SpecDecodeCostBreakdownRecord.num_partial_accepted_per_req` is `null`/`None` when
  the worker has not populated staged partial verification data. Downstream must not
  treat missing data as “zero partial accepts”; scheduler stats use `0` only for
  the legacy bridge when the list is absent.

## Hierarchical (staged) verification metrics
- **Three per-request series** in the cost record are intentionally different:
  - `num_intermediate_verified_tokens_per_req`: stage-0 (inter-verified) span length
    per request after origin collapse (**sum** over expanded rows mapped to that origin).
  - `num_intermediate_accepted_per_req`: tokens merged in intermediate acceptance
    rounds, summed over rounds; origin collapse is **sum** over expanded rows.
  - `num_partial_accepted_per_req`: effective provisional draft length per origin
    **immediately before target verification** — **winner-aware** collapse (first
    expanded row per origin, matching draft tensor collapse), **not** the sum of
    `len(prefix)` across mutually exclusive family branches.
- **Paper / figures** should use the same definitions as this note for
  `num_partial_accepted` (full pre-target provisional state), not the narrower
  “intermediate-accepted token count only” unless you rename the field.

## Cost timing overlap (nested attribution)
- `draft_forward_time_ms` covers the whole `propose_draft_token_ids()` path and is
  **inclusive** of hierarchical work.
- `partial_verification_time_ms` and `intermediate_verification_time_ms` attribute
  time **inside** that path; they are **not** disjoint wall-clock partitions.
  Do **not** naively sum `draft_forward + partial + intermediate` and expect a
  meaningful total wall time (the sum can exceed actual elapsed time).
- `intermediate_verification_time_ms` is recorded on the cost JSONL / transport
  only; the scheduler Prometheus bridge maps **`partial_verification_time_ms`**
  to `partial_verification_time_sec` and does **not** fold intermediate time into
  `SpecDecodingStats` today.

## Family metadata (`spec_decode_family_metadata.*.jsonl`)
- Optional pivot expand/collapse events may set `family_stage` to provisional
  literals such as `expand` / `collapse`. Consumers should treat unknown stage
  strings as opaque; a future schema may use stage-centric labels.

## `spec_decode_profile_max_steps`
- Once `max_steps` completed records have been finalized, `begin_step` and all
  emits are no-ops for later steps (no extra metadata row past the cap).

## Transitional compatibility notes
- Legacy `ModelRunnerOutput.spec_decode_cost_breakdown` is still written from the
  new transport to maintain compatibility.
- Scheduler consumes new compact transport when available, with legacy fallback.

## Offline parsing quick start
1. Load all JSONL files with pandas `read_json(..., lines=True)`.
2. Group by `profile_writer_id` and `spec_profile_step_id` for batch-level plots.
3. Join scheduler request metadata with cost breakdown on writer/step for
   per-step acceptance-vs-cost analyses.
4. For request trajectories, additionally join on `req_id`.
