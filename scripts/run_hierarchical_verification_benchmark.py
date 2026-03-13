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
  python scripts/run_hierarchical_verification_benchmark.py \\
      --model facebook/opt-125m \\
      --speculative-method ngram \\
      --num-speculative-tokens 4 \\
      --compression-ratios 1.0,0.75,0.5,0.25 \\
      --full-verification-interval 4 \\
      --num-prompts 4 \\
      --max-new-tokens 64
"""

import argparse
import time
from typing import List

from vllm.config import SpeculativeConfig
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
        choices=["ngram", "draft_model"],
        help="Speculative decoding method. "
        "For draft_model, you must also provide --draft-model.",
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
    return parser.parse_args()


def build_speculative_config(
    args: argparse.Namespace, compression_ratio: float
) -> SpeculativeConfig:
    """Create SpeculativeConfig for a given compression_ratio."""
    if args.speculative_method == "ngram":
        method = "ngram"
        model_field = "[ngram]"
        draft_load_config = None
    else:
        method = "draft_model"
        if not args.draft_model:
            raise ValueError(
                "speculative-method=draft_model requires --draft-model to be set."
            )
        model_field = args.draft_model
        draft_load_config = None

    return SpeculativeConfig(
        model=model_field,
        method=method,
        num_speculative_tokens=args.num_speculative_tokens,
        hierarchical_verification=True,
        compress_method="random",
        compression_ratio=compression_ratio,
        full_verification_interval=args.full_verification_interval,
        draft_load_config=draft_load_config,
    )


def run_once_for_ratio(
    args: argparse.Namespace, compression_ratio: float
) -> None:
    """Run a short generation benchmark for a single compression_ratio."""
    global _REJECTION_SAMPLER_TIME_SEC
    _REJECTION_SAMPLER_TIME_SEC = 0.0

    spec_cfg = build_speculative_config(args, compression_ratio)

    # NOTE: disable_log_stats=False to allow SpecDecodingLogging to print full
    # cost breakdown (draft/compression/partial/full) to logs.
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,  # type: ignore[arg-type]
        seed=0,
        speculative_config=spec_cfg,
        disable_log_stats=False,
    )

    prompts: List[str] = [args.prompt for _ in range(args.num_prompts)]
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - t0

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
    print(f"num_prompts       = {len(prompts)}")
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

    print(
        "Running hierarchical speculative verification benchmark with "
        f"compression_ratios={ratios} ..."
    )

    for r in ratios:
        run_once_for_ratio(args, r)


if __name__ == "__main__":
    main()

