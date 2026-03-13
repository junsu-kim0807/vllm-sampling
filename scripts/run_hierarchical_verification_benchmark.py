#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark hierarchical speculative verification with compressed KV cache.

This script:
- Runs simple generations with speculative decoding enabled.
- Sweeps over compression_ratio values.
- Measures:
  - End-to-end wall time and tokens-per-output (TPO).
  - Time spent inside the RejectionSampler (verification / sampling).
- Relies on the existing SpecDecodingLogging to print the detailed
  cost breakdown (draft/compression/partial/full) via standard logging.

Prerequisites:
- vLLM v1 stack must be importable from this repo.
- A GPU environment with the target model available.
- This script does not install dependencies (e.g., torch).

Example:
  # Sweep only compression ratios at a fixed batch size.
  python scripts/run_hierarchical_verification_benchmark.py \\
      --model facebook/opt-125m \\
      --speculative-method ngram \\
      --num-speculative-tokens 4 \\
      --compression-ratios 1.0,0.75,0.5,0.25 \\
      --full-verification-interval 4 \\
      --num-prompts 4 \\
      --max-new-tokens 64

  # Additionally sweep over batch sizes (per compression ratio).
  python scripts/run_hierarchical_verification_benchmark.py \\
      --model facebook/opt-125m \\
      --speculative-method ngram \\
      --num-speculative-tokens 4 \\
      --compression-ratios 1.0,0.5 \\
      --batch-sizes 1,4,16,32 \\
      --full-verification-interval 4 \\
      --max-new-tokens 64
"""

import argparse
import time
from typing import Any, List
from vllm.entrypoints.llm import LLM
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.rejection_sampler import RejectionSampler


_REJECTION_SAMPLER_TIME_SEC: float = 0.0


def _patch_rejection_sampler_timing() -> None:
    """Monkey-patch RejectionSampler.forward to accumulate wall time."""
    global _REJECTION_SAMPLER_TIME_SEC

    if getattr(RejectionSampler.forward, "_is_timed_wrapper", False):
        # Already patched.
        return

    orig_forward = RejectionSampler.forward

    def timed_forward(self, *args, **kwargs):  # type: ignore[override]
        global _REJECTION_SAMPLER_TIME_SEC
        t0 = time.perf_counter()
        out = orig_forward(self, *args, **kwargs)
        _REJECTION_SAMPLER_TIME_SEC += time.perf_counter() - t0
        return out

    timed_forward._is_timed_wrapper = True  # type: ignore[attr-defined]
    RejectionSampler.forward = timed_forward  # type: ignore[assignment]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark hierarchical speculative verification "
        "with compressed KV cache."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Target model name or path (for the main model).",
    )
    parser.add_argument(
        "--speculative-method",
        type=str,
        default="ngram",
        choices=["ngram", "draft_model", "eagle3"],
        help="Speculative decoding method. "
        "For draft_model or eagle3, you must also provide --draft-model.",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        default=None,
        help="Draft model name/path when using speculative-method=draft_model.",
    )
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        default=4,
        help="Number of speculative tokens for speculative decoding.",
    )
    parser.add_argument(
        "--compression-ratios",
        type=str,
        default="1.0,0.5",
        help="Comma-separated list of compression ratios to benchmark, "
        "e.g. '1.0,0.75,0.5,0.25'. 1.0 means no compression (full KV).",
    )
    parser.add_argument(
        "--full-verification-interval",
        type=int,
        default=4,
        help="Interval for full KV verification steps when hierarchical "
        "verification is enabled.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=4,
        help="Number of prompts to generate per compression_ratio.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default=None,
        help=(
            "Optional comma-separated list of batch sizes (number of prompts) "
            "to benchmark per compression_ratio, e.g. '1,4,16,32'. "
            "If not provided, uses --num-prompts as a single batch size."
        ),
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The quick brown fox jumps over the lazy dog.",
        help="Base prompt text to replicate num-prompts times.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum number of new tokens to generate per request.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for the LLM.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Model dtype for LLM (e.g., float16, bfloat16, auto).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. 0.0 corresponds to greedy decoding.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p for sampling.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=-1,
        help="Top-k for sampling (-1 for no top-k).",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help=(
            "Target fraction of GPU memory to use for KV cache, "
            "passed through to LLM(gpu_memory_utilization=...). "
            "Use a smaller value if you see 'Free memory ... is less than "
            "desired GPU memory utilization' errors."
        ),
    )
    return parser.parse_args()


def build_speculative_config(
    args: argparse.Namespace, compression_ratio: float
) -> dict[str, Any]:
    """Create a dict-like speculative_config for a given compression_ratio.

    We intentionally return a plain dict here because EngineArgs.create_speculative_config
    expects a mapping (e.g. the result of parsing CLI / JSON), not a SpeculativeConfig
    instance. The engine will construct the SpeculativeConfig model internally.
    """
    # Map high-level --speculative-method flag to SpeculativeConfig fields.
    if args.speculative_method == "ngram":
        method = "ngram"
        model_field = "[ngram]"
        base_cfg: dict[str, Any] = {
            "model": model_field,
            "method": method,
            "num_speculative_tokens": args.num_speculative_tokens,
        }
    elif args.speculative_method == "draft_model":
        method = "draft_model"
        if not args.draft_model:
            raise ValueError(
                "speculative-method=draft_model requires --draft-model to be set."
            )
        base_cfg = {
            "model": args.draft_model,
            "method": method,
            "num_speculative_tokens": args.num_speculative_tokens,
        }
    elif args.speculative_method == "eagle3":
        # For EAGLE3, the user must provide a speculator model via --draft-model.
        # See docs/features/speculative_decoding/eagle.md for examples.
        method = "eagle3"
        if not args.draft_model:
            raise ValueError(
                "speculative-method=eagle3 requires --draft-model to be set "
                "to the Eagle3 speculator model."
            )
        base_cfg = {
            "model": args.draft_model,
            "method": method,
            "num_speculative_tokens": args.num_speculative_tokens,
            # Common Eagle3 setups also specify draft_tensor_parallel_size,
            # but we let the engine infer it from tensor_parallel_size here.
        }
    else:
        raise ValueError(f"Unknown speculative-method: {args.speculative_method!r}")

    # If compression_ratio <= 0, treat this as a baseline config without
    # hierarchical verification / compression (pure speculative decoding).
    if compression_ratio <= 0.0:
        return base_cfg

    # Otherwise, enable hierarchical verification with the given compression_ratio.
    hv_cfg = {
        "hierarchical_verification": True,
        "compress_method": "random",
        "compression_ratio": compression_ratio,
        "full_verification_interval": args.full_verification_interval,
    }
    return {**base_cfg, **hv_cfg}


def run_once_for_ratio(
    args: argparse.Namespace,
    llm: LLM,
    compression_ratio: float,
    batch_size: int,
) -> None:
    """Run a short generation benchmark for a single (compression_ratio, batch_size)."""
    global _REJECTION_SAMPLER_TIME_SEC
    _REJECTION_SAMPLER_TIME_SEC = 0.0

    prompts: List[str] = [args.prompt for _ in range(batch_size)]
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - t0
    
    llm.llm_engine.do_log_stats()
    
    total_generated_tokens = 0
    for out in outputs:
        # Sum over all output sequences for robustness; usually 1.
        for seq in out.outputs:
            total_generated_tokens += len(seq.token_ids)

    tpo = (
        elapsed / total_generated_tokens if total_generated_tokens > 0 else float("inf")
    )

    print("=" * 80)
    print(f"compression_ratio = {compression_ratio:.4f}")
    print(f"batch_size        = {len(prompts)}")
    print(f"total_new_tokens  = {total_generated_tokens}")
    print(f"end_to_end_time_s = {elapsed:.4f}")
    print(f"end_to_end_TPO_s  = {tpo:.6f}  # seconds per generated token")
    print(
        f"rejection_sampler_time_s = {_REJECTION_SAMPLER_TIME_SEC:.6f} "
        f"({(_REJECTION_SAMPLER_TIME_SEC/elapsed*100.0 if elapsed > 0 else 0.0):.2f}% of end-to-end)"
    )
    print(
        "Note: Detailed speculative cost breakdown "
        "(draft/compression/partial/full times, acceptance lengths) "
        "is logged by SpecDecodingLogging to the standard logger."
    )


def main() -> None:
    args = parse_args()
    _patch_rejection_sampler_timing()

    ratios = [float(x) for x in args.compression_ratios.split(",") if x.strip()]
    if args.batch_sizes:
        batch_sizes = [
            int(x) for x in args.batch_sizes.split(",") if x.strip()
        ]
    else:
        batch_sizes = [args.num_prompts]

    print(
        "Running hierarchical speculative verification benchmark with "
        f"compression_ratios={ratios}, batch_sizes={batch_sizes} ..."
    )

    for r in ratios:
        # Create a fresh SpeculativeConfig & LLM per compression_ratio so that
        # we isolate effects of compression from other factors.
        spec_cfg = build_speculative_config(args, r)
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype,  # type: ignore[arg-type]
            seed=0,
            # EngineArgs expects speculative_config as a dict-like structure.
            speculative_config=spec_cfg,
            gpu_memory_utilization=args.gpu_memory_utilization,
            disable_log_stats=False,
        )
        print("=" * 80)
        print(f"### compression_ratio = {r:.4f}")
        for bs in batch_sizes:
            run_once_for_ratio(args, llm, r, bs)
            


if __name__ == "__main__":
    main()

