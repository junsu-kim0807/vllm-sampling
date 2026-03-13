# Hierarchical Speculative Verification with Compressed KV

This document describes the **hierarchical verification** extension for vLLM v1
speculative decoding, implemented in this branch. It introduces:

- A two-level verification scheme (partial → full).
- KV cache compression (token-level and block-level).
- Detailed cost breakdown metrics and a benchmarking script.

The design is entirely **backward compatible** with existing speculative
decoding flows. When `hierarchical_verification=False` (default), the new code
paths are inactive.

## 1. High-level idea

Speculative decoding normally verifies draft tokens using the **full KV cache**
of the target model at every step.

Hierarchical verification instead:

1. Performs one or more **partial verification** steps using a
   **compressed KV cache** (cheaper attention).
2. Periodically performs a **full verification** step using the **full KV
   cache** to re-validate or refresh the state.

The trade-off:

- Partial verification reduces per-step verification cost, at some risk of
  more rejections or shorter accepted prefixes.
- Full verification guarantees correctness but is more expensive.

The frequency of full verification is controlled by
`SpeculativeConfig.full_verification_interval`.

## 2. New configuration fields

File: `vllm/config/speculative.py`

```python
# Compression method for hierarchical verification (partial KV cache).
# - "random": token-level compression with compaction (full -> compressed buffer).
# - "block_random": block-level compression without KV memory copy
#   (only block_table/seq_lens subsampling).
HierarchicalVerificationCompressMethod = Literal["random", "block_random"]

@config
class SpeculativeConfig:
    ...
    # Hierarchical verification: partial (compressed) KV verification first,
    # then full KV verification for partially verified tokens.
    hierarchical_verification: bool = False
    """If True, use partial (compressed) KV cache for initial verification
    steps, then full KV cache to verify partially accepted tokens."""

    compress_method: HierarchicalVerificationCompressMethod = "random"
    """Method to compress KV cache for partial verification."""

    compression_ratio: float = Field(default=0.5, gt=0, le=1)
    """Ratio of KV cache to use in partial verification (e.g. 0.5 = 50%)."""

    full_verification_interval: int = Field(default=1, ge=1)
    """When hierarchical_verification is True, use full KV cache every this
    many steps; otherwise use partial (compressed) KV cache. Must be >= 1."""
```

The `compute_hash()` method of `SpeculativeConfig` was updated to include:

- `hierarchical_verification`
- `compress_method`
- `compression_ratio`
- `full_verification_interval`

so that changing these values invalidates compilation caches appropriately.

## 3. KV compression module

File: `vllm/v1/spec_decode/kv_compression.py`

This module implements two kinds of compression:

### 3.1 Token-level compression (method=`"random"`)

- `random_compression_mask(...)` / `random_compression_indices(...)`:

  - Select a random subset of token positions (same indices for all layers).

- `get_compression_indices_batch(...)`:

  - Public helper to compute a batch-wide list of kept token indices, or `None`
    when `compression_ratio >= 1.0`.

- `build_compressed_kv_caches(...)`:

  - For each per-layer KV cache tensor in `full_kv_caches`:
    - Gathers only the slots referenced by the kept token indices
      (via `slot_mapping`).
    - Packs these into a **new compressed KV buffer** with fewer blocks:
      `[... num_blocks_compressed, block_size, ...]`.

- `CompressedKVMetadata` and `build_compressed_kv_metadata(...)`:

  - Build a synthetic `block_table` and `seq_lens` for the compressed buffer.
  - `seq_lens` is per-request compressed length (tokens).
  - `block_table` describes which compressed blocks each request will use.
  - `num_compressed_slots` is the total number of compressed KV slots.

This path requires **actual memory copy / compaction** from the full KV cache
into a smaller temporary buffer.

### 3.2 Block-level compression (method=`"block_random"`)

When `compress_method="block_random"`, we compress at the **block level** and
do **not** move KV memory:

- `random_block_indices_for_req(...)`:

  - For a single request, pick `compression_ratio` fraction of blocks by index.

- `build_block_level_compression_view(block_table_np, num_blocks_per_row, ...)`:

  - Given:
    - `block_table_np`: `[num_reqs, max_num_blocks]` block IDs.
    - `num_blocks_per_row`: `[num_reqs]` valid block counts per request.
  - Returns:
    - `compressed_block_table`: `[num_reqs, max_kept]` subset of block IDs,
      padded with `-1`.
    - `compressed_blocks_per_row`: `[num_reqs]` kept block counts.

At runtime:

- We convert these numpy arrays to torch tensors and update:
  - `CommonAttentionMetadata.seq_lens` to reflect `kept_blocks * block_size`.
  - `CommonAttentionMetadata.block_table_tensor` to the compressed view.
- The **underlying KV cache tensors (`kv_caches`) are left unchanged**.
  - Attention simply “sees” fewer blocks when computing.

## 4. Hierarchical verification in `GPUModelRunner`

File: `vllm/v1/worker/gpu_model_runner.py`

### 4.1 New state and helpers

- In `GPUModelRunner.__init__`:

  ```python
  self._hierarchical_verification_step: int = 0
  ```

  - Per-runner counter: incremented every speculative forward step when
    hierarchical verification is enabled.

- New helper methods:

  ```python
  def _get_ordered_kv_layer_names(self) -> list[str]:
      """Return KV layer names in the same order as self.kv_caches (for swap)."""
      ...

  def _swap_kv_caches_for_partial(
      self, compressed_caches: list[torch.Tensor]
  ) -> None:
      """Temporarily bind compressed KV caches to attention layers."""
      ...

  def _restore_kv_caches(self) -> None:
      """Restore full KV caches on attention layers after partial verification."""
      ...
  ```

  - These map from layer names to `self.kv_caches` and temporarily replace
    `layer.kv_cache` tensors with compressed ones, then restore them.

### 4.2 Partial vs full step selection

Inside `execute_model(...)`, before building attention metadata:

```python
use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
...
spec_config = self.speculative_config
is_partial_step = False
compressed_kv_caches: list[torch.Tensor] | None = None
compressed_kv_metadata: CompressedKVMetadata | None = None

if (
    use_spec_decode
    and spec_config is not None
    and getattr(spec_config, "hierarchical_verification", False)
    and len(self.kv_caches) > 0
):
    interval = getattr(spec_config, "full_verification_interval", 1)
    is_partial_step = (self._hierarchical_verification_step % interval != 0)
    if is_partial_step:
        seed = getattr(self.vllm_config.model_config, "seed", 0)
        if getattr(spec_config, "compress_method", "random") == "random":
            # Token-level compression path (copies into compressed KV buffers)
            ...
        elif getattr(spec_config, "compress_method", "random") == "block_random":
            # Block-level compression path (no KV copy)
            ...
        else:
            is_partial_step = False
```

- When `is_partial_step` is `True`:
  - `"random"`:
    - Build compressed KV buffers per layer (`build_compressed_kv_caches`).
    - Build `CompressedKVMetadata` for the compressed buffer.
  - `"block_random"`:
    - Use `BlockTable` CPU arrays (`block_table.np`, `num_blocks_per_row`).
    - Call `build_block_level_compression_view(...)`.
    - Build `seq_lens_comp = compressed_blocks_per_row * block_size`.
    - Instantiate `CompressedKVMetadata` where:
      - `seq_lens` = `seq_lens_comp.cpu().numpy()`.
      - `block_table` = `compressed_block_table_np`.
      - `num_compressed_slots` = total compressed tokens.

### 4.3 Attention metadata override

Still in `execute_model`, after computing base metadata:

```python
block_table_gid_0 = _get_block_table(0)
slot_mapping_gid_0 = slot_mappings[0]

cm_base = CommonAttentionMetadata(
    query_start_loc=self.query_start_loc.gpu[: num_reqs_padded + 1],
    query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs_padded + 1],
    seq_lens=self.seq_lens.gpu[:num_reqs_padded],
    _seq_lens_cpu=self.seq_lens.cpu[:num_reqs_padded],
    _num_computed_tokens_cpu=self.input_batch.num_computed_tokens_cpu_tensor[
        :num_reqs_padded
    ],
    num_reqs=num_reqs_padded,
    num_actual_tokens=num_tokens_padded,
    max_query_len=max_query_len,
    max_seq_len=max_seq_len,
    block_table_tensor=block_table_gid_0,
    slot_mapping=slot_mapping_gid_0,
    causal=True,
)
if compressed_kv_metadata is not None:
    # Override seq_lens and block_table for compressed view.
    seq_lens_comp = torch.zeros(
        num_reqs_padded, dtype=torch.int32, device=self.device
    )
    seq_lens_comp[:num_reqs] = torch.as_tensor(
        compressed_kv_metadata.seq_lens,
        dtype=torch.int32,
        device=self.device,
    )
    cm_base.seq_lens = seq_lens_comp

    n_comp_blocks = compressed_kv_metadata.block_table.shape[1]
    block_table_comp = torch.full(
        (num_reqs_padded, n_comp_blocks),
        -1,
        dtype=torch.int32,
        device=self.device,
    )
    block_table_comp[:num_reqs] = torch.as_tensor(
        compressed_kv_metadata.block_table,
        dtype=torch.int32,
        device=self.device,
    )
    cm_base.block_table_tensor = block_table_comp
    cm_base.max_seq_len = int(compressed_kv_metadata.num_compressed_slots)
```

- For both compression modes:
  - Attention backends see a `CommonAttentionMetadata` where:
    - `seq_lens` and `block_table_tensor` describe the **compressed view**.
  - In the token-level path, this view points into the **compressed KV buffer**.
  - In the block-level path, the view describes a **subset of the original
    blocks**, so no KV copy is required.

### 4.4 Timing and stats (partial vs full)

Forward timing in `execute_model`:

```python
clear_kv_metadata = self.speculative_config is None
partial_verification_time_sec = 0.0
full_verification_time_sec = 0.0
with (
    set_forward_context(..., compressed_kv_caches=compressed_kv_caches,
                        compressed_kv_metadata=compressed_kv_metadata),
    record_function_or_nullcontext("gpu_model_runner: forward"),
    self.maybe_get_kv_connector_output(...) as kv_connector_output,
):
    if compressed_kv_caches is not None:
        self._swap_kv_caches_for_partial(compressed_kv_caches)
    try:
        t_forward_start = time.perf_counter()
        model_output = self._model_forward(...)
        elapsed_forward = time.perf_counter() - t_forward_start
        # Attribute forward time either to partial or full verification,
        # depending on whether this step is using compressed KV.
        if (
            use_spec_decode
            and spec_config is not None
            and getattr(spec_config, "hierarchical_verification", False)
            and is_partial_step
        ):
            partial_verification_time_sec = elapsed_forward
        else:
            full_verification_time_sec = elapsed_forward
    finally:
        if compressed_kv_caches is not None:
            self._restore_kv_caches()

    if (
        use_spec_decode
        and spec_config is not None
        and getattr(spec_config, "hierarchical_verification", False)
    ):
        self._hierarchical_verification_step += 1
```

Later, in `sample_tokens(...)` we construct:

```python
spec_decode_cost_breakdown = None
if (
    use_spec_decode
    and spec_config is not None
    and spec_config.hierarchical_verification
):
    num_reqs = len(req_ids_output_copy)
    spec_decode_cost_breakdown = SpecDecodeCostBreakdown(
        draft_time_sec=hierarchical_draft_time_sec,
        compression_time_sec=hierarchical_compression_time_sec,
        partial_verification_time_sec=partial_verification_time_sec,
        full_verification_time_sec=full_verification_time_sec,
        num_partial_accepted_per_req=[0] * num_reqs,
    )
```

This is then passed to the scheduler as part of `ModelRunnerOutput`.

## 5. Scheduler integration and metrics

### 5.1 Cost breakdown data path

- File: `vllm/v1/outputs.py`

  ```python
  @dataclass
  class SpecDecodeCostBreakdown:
      draft_time_sec: float = 0.0
      compression_time_sec: float = 0.0
      partial_verification_time_sec: float = 0.0
      full_verification_time_sec: float = 0.0
      num_partial_accepted_per_req: list[int] = field(default_factory=list)

  @dataclass
  class ModelRunnerOutput:
      ...
      spec_decode_cost_breakdown: SpecDecodeCostBreakdown | None = None
  ```

- File: `vllm/v1/core/sched/scheduler.py`

  - `Scheduler.update_from_output(...)` extracts
    `model_runner_output.spec_decode_cost_breakdown` and passes it into
    `make_spec_decoding_stats(...)`.

  - `make_spec_decoding_stats(...)`:

    ```python
    if cost_breakdown is not None and req_index is not None:
        num_partial = (
            cost_breakdown.num_partial_accepted_per_req[req_index]
            if req_index < len(cost_breakdown.num_partial_accepted_per_req)
            else 0
        )
        # Add batch-level times only for the first request in this batch.
        is_first_spec_req = spec_decoding_stats.num_drafts == 0
        spec_decoding_stats.observe_draft_with_cost_breakdown(
            num_draft_tokens=num_draft_tokens,
            num_accepted_tokens=num_accepted_tokens,
            draft_time_sec=cost_breakdown.draft_time_sec if is_first_spec_req else 0,
            compression_time_sec=(
                cost_breakdown.compression_time_sec if is_first_spec_req else 0
            ),
            partial_verification_time_sec=(
                cost_breakdown.partial_verification_time_sec
                if is_first_spec_req
                else 0
            ),
            num_partial_accepted_tokens=num_partial,
            full_verification_time_sec=(
                cost_breakdown.full_verification_time_sec if is_first_spec_req else 0
            ),
        )
    ```

  - Batch-level times (draft/compression/partial/full) are added **once per
    batch**, while per-request `num_partial_accepted_tokens` is added per
    request.

### 5.2 Logging

File: `vllm/v1/spec_decode/metrics.py`

- `SpecDecodingStats` has additional fields:

  ```python
  draft_time_sec: float = 0.0
  compression_time_sec: float = 0.0
  partial_verification_time_sec: float = 0.0
  num_partial_accepted_tokens: int = 0
  full_verification_time_sec: float = 0.0
  ```

- `SpecDecodingLogging.log(...)` aggregates and prints:

  ```python
  total_draft_time = np.sum(self.draft_time_sec)
  total_compression_time = np.sum(self.compression_time_sec)
  total_partial_verification_time = np.sum(self.partial_verification_time_sec)
  total_partial_accepted = np.sum(self.num_partial_accepted_tokens)
  total_full_verification_time = np.sum(self.full_verification_time_sec)
  ...
  partial_acceptance_length = (
      1 + total_partial_accepted / num_drafts if num_drafts > 0 else 0.0
  )

  if has_cost_breakdown:
      log_fn(
          "SpecDecoding cost breakdown: "
          "draft_time: %.4fs, compression_time: %.4fs, "
          "partial_verification_time: %.4fs, partial_acceptance_length: %.2f, "
          "full_verification_time: %.4fs, full_acceptance_length: %.2f",
          total_draft_time,
          total_compression_time,
          total_partial_verification_time,
          partial_acceptance_length,
          total_full_verification_time,
          mean_acceptance_length,
      )
  ```

These logs give a coarse-grained breakdown of how much time is spent in
drafting, compression, partial verification, and full verification over each
logging interval.

## 6. Benchmark script

File: `scripts/run_hierarchical_verification_benchmark.py`

This script benchmarks hierarchical speculative verification using the vLLM
`LLM` entrypoint.

Features:

- Sweeps over:
  - `compression_ratio` values (`--compression-ratios`).
  - Optional batch sizes (`--batch-sizes`), i.e. number of prompts.
- Measures:
  - End-to-end time per generation call.
  - Tokens per output (TPO, seconds per generated token).
  - Time spent **inside the RejectionSampler** (using a monkey-patch).
- Relies on `SpecDecodingLogging` to print the detailed cost breakdown.

Key arguments:

- `--model`: target model name/path.
- `--speculative-method`: `"ngram"` or `"draft_model"`.
- `--draft-model`: draft model name/path when using `draft_model` method.
- `--num-speculative-tokens`: number of speculative tokens.
- `--compression-ratios`: comma-separated list, e.g. `1.0,0.75,0.5`.
- `--full-verification-interval`: interval between full verification steps.
- `--num-prompts`: default batch size when `--batch-sizes` is not provided.
- `--batch-sizes`: optional comma-separated batch sizes, e.g. `1,4,16,32`.

Example usage:

```bash
# Sweep only compression ratios at a fixed batch size.
python scripts/run_hierarchical_verification_benchmark.py \
  --model facebook/opt-125m \
  --speculative-method ngram \
  --num-speculative-tokens 4 \
  --compression-ratios 1.0,0.75,0.5,0.25 \
  --full-verification-interval 4 \
  --num-prompts 4 \
  --max-new-tokens 64

# Additionally sweep over batch sizes (per compression ratio).
python scripts/run_hierarchical_verification_benchmark.py \
  --model facebook/opt-125m \
  --speculative-method ngram \
  --num-speculative-tokens 4 \
  --compression-ratios 1.0,0.5 \
  --batch-sizes 1,4,16,32 \
  --full-verification-interval 4 \
  --max-new-tokens 64
```

For each `(compression_ratio, batch_size)` combination, the script prints:

- `compression_ratio`
- `batch_size`
- `total_new_tokens`
- `end_to_end_time_s`
- `end_to_end_TPO_s` (seconds per generated token)
- `rejection_sampler_time_s` and its percentage of end-to-end time.

This allows comparing:

- token-level vs block-level compression (`compress_method`),
- different compression ratios,
- and different batch sizes,

under realistic speculative decoding workloads.

