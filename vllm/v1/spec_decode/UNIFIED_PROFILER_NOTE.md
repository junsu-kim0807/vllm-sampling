# Unified Speculative Decode Profiler

## Architecture

The profiler has two layers:

- **Layer A (Online)**: Low-overhead, always-on. Captures stage wall-time, shape metadata (batch/family/seq dimensions), memory allocation snapshots (allocated + reserved, before/after/peak), KV-load work proxy, and KV working-set proxy.
- **Layer B (Deep Kernel)**: Expensive, sampled steps only. Wraps each stage scope with `torch.profiler.profile`, classifies captured raw CUDA events by kernel category, splits GEMMs into memory-dominated vs compute-dominated, and tags each category with a confidence level.

## Modes

| Mode              | Online stage cost | Shape + memory | Kernel breakdown | Metadata |
|-------------------|:-----------------:|:--------------:|:----------------:|:--------:|
| `disabled`        |                   |                |                  |          |
| `stage_cost`      | ✓                 |                |                  | ✓        |
| `shape_memory`    | ✓                 | ✓              |                  | ✓        |
| `kernel_breakdown`| ✓                 |                | ✓                | ✓        |
| `all`             | ✓                 | ✓              | ✓                | ✓        |

## Stage taxonomy

Stages are defined at the **actual model forward invocation point** (not wrapper boundaries) to prevent double-counting:

- `draft_forward` — draft model forward call
- `intermediate_verify` — intermediate verifier forward call
- `target_verify` — target model forward (only on speculative verification path; `forward_reason` tag distinguishes from prefill/decode)
- `expand_collapse` — family expansion + winner selection + row collapse
- `reject_sample` — rejection sampling / acceptance decision
- `bookkeeping` — metadata build, scheduler bridge, output construction (tracks sub-timings: `metadata_build_ms`, `scheduler_bridge_ms`, `output_pack_ms`)

## Join keys

Records across JSONL files can be joined on: `profile_writer_id + step_id + stage_name + stage_invocation_idx`. Request-level joins use `req_id`.

## Weight-load attribution guide

### Why `weight_load_time_ms` cannot be directly measured

Weight loading happens **inside** GEMM kernels. Attention kernels (e.g., FlashAttention, PagedAttention) have recognizable names, but parameter-side memory access is fused into the same GEMM that does the computation. There is no kernel-level boundary between "reading weights from HBM" and "computing the matrix multiply."

### What `gemm_memory_dominated_time_ms` actually measures

This field is a **memory-bandwidth-dominated GEMM proxy**. It captures GEMMs whose arithmetic intensity (`FLOPs / bytes_accessed`) is below a configurable device-specific threshold (default 50.0 for A100). These GEMMs spend most of their runtime waiting on memory bandwidth.

**Critical**: Memory-dominated GEMMs include **both** parameter-side streaming **and** activation-side data movement. The two cannot be separated at the kernel level.

**In figure captions and papers, label this as:**
- "memory-bandwidth-dominated GEMM time" ✓
- "GEMM memory proxy" ✓
- "weight load time" ✗ (incorrect — overattributes to weights)
- "weight streaming time" ✗ (incorrect — same reason)

### Interpretation matrix

| `gemm_memory_dominated` high | `kv_sensitive_attention` high | Likely cause |
|:---:|:---:|:---|
| ✓ | ✓ | Overall memory-bandwidth bottleneck (both weights and KV cache) |
| ✓ |   | Weight-streaming dominated (small batch, large model) |
|   | ✓ | KV-cache access dominated (long sequences, many expansion rows) |
|   |   | Compute-bound or communication-bound |

### Confidence tags

- **`direct`**: Kernel name unambiguously maps to category (e.g., `flash_attn_fwd` → attention)
- **`derived`**: Inferred from operator call-stack context (e.g., GEMM under MLP module)
- **`proxy`**: Estimated from heuristics (e.g., arithmetic intensity threshold for GEMM split)

## KV working-set proxy

### Definition

`kv_working_set_pages` counts the number of **distinct** KV cache pages that attention *should have* read during a stage, derived from `block_table[active_rows, :ceil(context_len / page_size)]`.

### Accuracy caveat

This is a **software-level proxy**, not a hardware trace. It assumes standard full-context attention. The proxy diverges from ground truth when:

- **Compression** is active (fewer pages actually read)
- **Sink + recent view** patterns skip middle pages
- **Provisional frontier** limits the context window
- **Sparse attention** patterns skip arbitrary pages

Label this as "working-set proxy" in all figures and analysis. Do not claim it represents actual hardware cache line touches.

### Diagnostic value

- "KV live footprint barely changed but working set doubled" → expansion duplicated attention over the same pages
- "Both footprint and working set grew proportionally" → expansion allocated new KV pages, inflating both

## Memory inflation tracking

For each stage invocation, six values are recorded:

| Field | Source |
|-------|--------|
| `alloc_before_bytes` | `torch.cuda.memory_allocated()` before stage |
| `alloc_after_bytes` | `torch.cuda.memory_allocated()` after stage |
| `alloc_peak_bytes` | `torch.cuda.max_memory_allocated()` after stage |
| `reserved_before_bytes` | `torch.cuda.memory_reserved()` before stage |
| `reserved_after_bytes` | `torch.cuda.memory_reserved()` after stage |
| `reserved_peak_bytes` | `torch.cuda.max_memory_reserved()` after stage |

`torch.cuda.reset_peak_memory_stats()` is called before each stage to capture within-stage peaks accurately.

**Why both allocated and reserved?** PyTorch's caching allocator may hold reserved memory blocks that don't appear in `memory_allocated()`. Expansion-induced allocation spikes may be invisible in `allocated` if the allocator reuses reserved blocks. `reserved_peak` captures the true high-water mark.

## Per-request token semantics

- `num_intermediate_verified_tokens_per_req`: **Sum** over expanded rows per origin
- `num_intermediate_accepted_tokens_per_req`: **Sum** over expanded rows per origin
- `num_partial_accepted_per_req`: **Winner-aware** (first expanded row per origin)

## First-draft top-k in `request_metadata`

Per request and per speculative step, optional fields describe the drafter’s **first speculative position** distribution:

- `first_draft_topk_token_ids`, `first_draft_topk_confidences`, `first_draft_topk_k`
- `first_draft_topk_source`: `"profile_first_draft_topk"` when aligned data exists, or `"unavailable"` when not.

**Stable cache:** Values come from a step-scoped **profile cache** (`profile_first_draft_topk_*` on `SpecDecodeBaseProposer`), not from `last_root_topk_info` (pivot scratch, cleared/overwritten by inner rounds). `clear_draft_probs()` does **not** clear the profile cache; the runner resets it once per step after profiler `begin_step` and before the first draft proposal.

**Row alignment:** The cache stores `RootTopKInfo` plus a **req_id snapshot** taken when the cache is filled. At emission, the runner builds a **`req_id → row index` map once** from that snapshot and attaches top-k per output `req_id` in **O(1)** per request. `last_root_topk_info` without a matching snapshot is **never** used for these fields (avoids wrong row alignment after pivot/bundle remaps).

**Probabilities:** On the Eagle/draft base path, confidences use **`logsumexp` + `topk` on logits** (equivalent softmax mass on the reported top-k without materializing full softmax).

**Coverage:** Populated for **EAGLE**, **draft_model**, **pivot** (wrapper freeze after root proposal), and **adaptive_spechive** (delegates to the same draft proposer). **Medusa**, **ngram**, **suffix**, and **extract_hidden_states** do not fill the cache yet; records use `first_draft_topk_source="unavailable"`.

## Offline analysis recipe

1. Join `stage_shape` + `stage_memory` on `(step_id, stage_name)`
2. Correlate `expansion_pct` vs `reserved_peak_bytes` vs `kv_working_set_pages`
3. On sampled steps: join `stage_kernel` to see `gemm_memory_dominated_time_ms` vs `kv_sensitive_attention_time_ms`
4. Build scatter plots: expansion → memory inflation → kernel time decomposition

## Kernel category map

Version: `v1`

Direct classifications: attention (flash_attn, paged_attention, flashinfer, sdpa, attn_fwd, fmha, cutlass_fmha), communication (nccl, all_reduce, all_gather, reduce_scatter, broadcast), sampling (top_k, top_p, sample, multinomial, rejection_sample), normalization (layernorm, rmsnorm, rms_norm), activation (silu_mul, gelu, relu, swiglu).

GEMM classification: context-based (call stack) → derived confidence; arithmetic-intensity-based → proxy confidence.
