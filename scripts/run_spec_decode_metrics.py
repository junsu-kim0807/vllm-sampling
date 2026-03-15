#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Run vanilla speculative decoding (draft_model) on AIME 2025 and CodeElo.

When --profile_time is set, sets VLLM_SPEC_PROFILE_TIME=1 so that vLLM
measures draft and verification time, then reports:
  - average draft time, average verification time
  - average acceptance rate and position-wise acceptance rate

Output: CSV and JSONL with one row per (target_model, dataset, batch_size).

NOTE: draft_time_s / verification_time_s in the output are only filled when
  (1) you pass --profile-time, and
  (2) get_metrics() returns the spec_decode_draft_time_seconds_total and
      spec_decode_verification_time_seconds_total counters (same process).
If you see null for those fields but the vLLM log shows "SpecDecoding cost
breakdown: draft_time: ...", the engine is measuring them; the script just
could not read them (e.g. disable_log_stats=True or metrics from another
process). Pass --profile-time and ensure disable_log_stats is False.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import time
from typing import Any

from vllm.v1.metrics.reader import Counter, Gauge, Vector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run speculative decoding (draft Qwen3-0.6B, target 4B/8B/30B-A3B) "
            "on AIME 2025 and CodeElo. With --profile_time, report average "
            "draft/verification time and acceptance rates."
        )
    )
    p.add_argument("--draft-model", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument(
        "--target-models",
        type=str,
        default=(
            "Qwen/Qwen3-4B,"
            "Qwen/Qwen3-8B,"
            "Qwen/Qwen3-30B-A3B"
        ),
        help="Comma-separated target model list.",
    )
    p.add_argument(
        "--tp-map",
        type=str,
        default=(
            "Qwen/Qwen3-4B=1,"
            "Qwen/Qwen3-8B=1,"
            "Qwen/Qwen3-30B-A3B=2"
        ),
        help="Comma-separated target_model=tp map.",
    )
    p.add_argument(
        "--datasets",
        type=str,
        default="aime25,codeelo",
        help="Comma-separated dataset keys. Supported: aime25, codeelo",
    )
    p.add_argument(
        "--batch-sizes",
        type=str,
        default="1",
        help="Comma-separated batch sizes.",
    )
    p.add_argument("--num-spec-tokens", type=int, default=3)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--dtype", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--enable-chunked-prefill", action="store_true")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--disable-log-stats", action="store_true")
    p.add_argument("--disable-custom-all-reduce", action="store_true")
    p.add_argument(
        "--max-samples-aime25",
        type=int,
        default=None,
        help="Optional cap for AIME25 samples.",
    )
    p.add_argument(
        "--max-samples-codeelo",
        type=int,
        default=None,
        help="Optional cap for CodeElo samples.",
    )
    p.add_argument(
        "--aime-max-new-tokens",
        type=int,
        default=256,
        help="Max new tokens for AIME25.",
    )
    p.add_argument(
        "--codeelo-max-new-tokens",
        type=int,
        default=1024,
        help="Max new tokens for CodeElo.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1)
    p.add_argument(
        "--warmup-iters",
        type=int,
        default=2,
    )
    p.add_argument(
        "--warmup-max-tokens",
        type=int,
        default=32,
    )
    p.add_argument(
        "--profile-time",
        action="store_true",
        help="Set VLLM_SPEC_PROFILE_TIME=1 and report average draft/verification time.",
    )
    p.add_argument(
        "--results-csv",
        type=str,
        default="spec_decode_metrics_results.csv",
    )
    p.add_argument(
        "--results-jsonl",
        type=str,
        default="spec_decode_metrics_results.jsonl",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def parse_csv_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_tp_map(raw: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in parse_csv_list(raw):
        if "=" not in item:
            raise ValueError(f"Invalid --tp-map item: {item}")
        k, v = item.split("=", 1)
        out[k.strip()] = int(v.strip())
    return out


AIME25_REPO = "opencompass/AIME2025"
CODEELO_REPO = "Qwen/CodeElo"


def load_dataset_split(repo: str):
    from datasets import load_dataset, concatenate_datasets

    if repo == AIME25_REPO:
        ds_i = load_dataset(repo, "AIME2025-I", split="test")
        ds_ii = load_dataset(repo, "AIME2025-II", split="test")
        ds = concatenate_datasets([ds_i, ds_ii])
        return ds, "test"

    last_err = None
    for split in ("test", "validation", "train"):
        try:
            return load_dataset(repo, split=split), split
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not load dataset {repo}: {last_err}") from last_err


def extract_first_present(
    example: dict[str, Any],
    keys: list[str],
    default: str = "",
) -> str:
    for key in keys:
        if key in example and example[key] is not None:
            return str(example[key])
    return default


def build_aime25_prompt(example: dict[str, Any]) -> str:
    problem = extract_first_present(
        example,
        ["problem", "question", "input", "prompt"],
    )
    return (
        "Solve the following AIME 2025 problem. "
        "Return only the final answer as a non-negative integer.\n\n"
        f"Problem:\n{problem}\n\n"
        "Final answer:"
    )


def build_codeelo_prompt(example: dict[str, Any]) -> str:
    title = extract_first_present(example, ["name", "title"], default="")
    description = extract_first_present(example, ["description"], default="")
    input_spec = extract_first_present(example, ["input"], default="")
    output_spec = extract_first_present(example, ["output"], default="")
    interaction = extract_first_present(example, ["interaction"], default="")
    note = extract_first_present(example, ["note"], default="")

    sections: list[str] = []
    if title:
        sections.append(f"Title:\n{title}")
    if description:
        sections.append(f"Problem:\n{description}")
    if input_spec:
        sections.append(f"Input Format:\n{input_spec}")
    if output_spec:
        sections.append(f"Output Format:\n{output_spec}")
    if interaction:
        sections.append(f"Interaction:\n{interaction}")
    if note:
        sections.append(f"Notes:\n{note}")

    body = "\n\n".join(sections)
    return (
        "Solve the following competitive programming problem. "
        "Output only the final C++17 solution code inside one markdown code block.\n\n"
        f"{body}\n\n"
        "Answer:"
    )


def get_dataset_prompts(dataset_key: str, args: argparse.Namespace) -> list[str]:
    if dataset_key == "aime25":
        ds, split = load_dataset_split(AIME25_REPO)
        rows = list(ds)
        if args.max_samples_aime25 is not None:
            rows = rows[: args.max_samples_aime25]
        prompts = [build_aime25_prompt(x) for x in rows]
        print(
            f"[dataset] {dataset_key}: repo={AIME25_REPO}, "
            f"split={split}, samples={len(prompts)}"
        )
        return prompts

    if dataset_key == "codeelo":
        ds, split = load_dataset_split(CODEELO_REPO)
        rows = list(ds)
        if args.max_samples_codeelo is not None:
            rows = rows[: args.max_samples_codeelo]
        prompts = [build_codeelo_prompt(x) for x in rows]
        print(
            f"[dataset] {dataset_key}: repo={CODEELO_REPO}, "
            f"split={split}, samples={len(prompts)}"
        )
        return prompts

    raise ValueError(f"Unsupported dataset key: {dataset_key}")


def get_dataset_max_new_tokens(dataset_key: str, args: argparse.Namespace) -> int:
    if dataset_key == "aime25":
        return args.aime_max_new_tokens
    if dataset_key == "codeelo":
        return args.codeelo_max_new_tokens
    raise ValueError(dataset_key)


def snapshot_metrics(llm) -> dict[str, Any]:
    """Aggregate get_metrics() into a single dict (sum over engines)."""
    out: dict[str, Any] = {}
    try:
        metrics = llm.get_metrics()
    except Exception:
        return out

    for metric in metrics:
        name = getattr(metric, "name", None)
        if not name:
            continue
        if isinstance(metric, Counter):
            out[name] = out.get(name, 0) + metric.value
        elif isinstance(metric, Gauge):
            # Sum gauges with same name (e.g. multi-engine time counters)
            out[name] = out.get(name, 0.0) + float(metric.value)
        elif isinstance(metric, Vector):
            if name not in out:
                out[name] = list(metric.values)
            else:
                cur = out[name]
                for i, v in enumerate(metric.values):
                    if i < len(cur):
                        cur[i] = cur[i] + v
                    else:
                        cur.append(v)
    return out


def metric_delta(
    after: dict[str, Any], before: dict[str, Any], name: str
) -> float | int | None:
    if name not in after and name not in before:
        return None
    return after.get(name, 0) - before.get(name, 0)


def vector_delta(
    after: dict[str, Any], before: dict[str, Any], name: str
) -> list[int]:
    a = after.get(name, []) or []
    b = before.get(name, []) or []
    n = max(len(a), len(b))
    return [(a[i] if i < len(a) else 0) - (b[i] if i < len(b) else 0) for i in range(n)]


SPEC_NUM_DRAFTS = "vllm:spec_decode_num_drafts"
SPEC_NUM_DRAFT_TOKENS = "vllm:spec_decode_num_draft_tokens"
SPEC_NUM_ACCEPTED = "vllm:spec_decode_num_accepted_tokens"
SPEC_ACCEPTED_PER_POS = "vllm:spec_decode_num_accepted_tokens_per_pos"
SPEC_DRAFT_TIME = "vllm:spec_decode_draft_time_seconds_total"
SPEC_VERIFICATION_TIME = "vllm:spec_decode_verification_time_seconds_total"
SPEC_DRAFT_VERIF_CHECKS = "vllm:spec_decode_draft_verification_checks_total"
SPEC_DRAFT_VERIF_MISMATCHES = "vllm:spec_decode_draft_verification_mismatches_total"


def chunked(prompts: list[str], batch_size: int):
    for i in range(0, len(prompts), batch_size):
        yield prompts[i : i + batch_size]


def apply_chat_template(tokenizer, prompts: list[str]) -> list[str]:
    out: list[str] = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        out.append(text)
    return out


def maybe_cuda_sync() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def run_warmup(
    llm: Any,
    tokenizer: Any,
    prompts: list[str],
    batch_size: int,
    warmup_iters: int,
    warmup_max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> None:
    if warmup_iters <= 0:
        return

    from vllm import SamplingParams

    warm = list(prompts[:batch_size])
    if not warm:
        return
    while len(warm) < batch_size:
        warm.append(warm[-1])

    warm = apply_chat_template(tokenizer, warm)
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=warmup_max_tokens,
    )
    print(f"[warmup] iters={warmup_iters}, batch_size={batch_size}, max_tokens={warmup_max_tokens}")
    for idx in range(warmup_iters):
        maybe_cuda_sync()
        _ = llm.generate(warm, sampling_params=sampling_params)
        maybe_cuda_sync()
        print(f"[warmup] done {idx + 1}/{warmup_iters}")


def measure_dataset(
    llm: Any,
    tokenizer: Any,
    prompts: list[str],
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    profile_time: bool,
    verbose: bool = False,
) -> dict[str, Any]:
    from vllm import SamplingParams

    batches = [
        apply_chat_template(tokenizer, batch)
        for batch in chunked(list(prompts), batch_size)
    ]
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_new_tokens,
    )

    before = snapshot_metrics(llm)
    total_output_tokens = 0

    maybe_cuda_sync()
    t0 = time.perf_counter()

    for batch_idx, prompt_batch in enumerate(batches, start=1):
        maybe_cuda_sync()
        outputs = llm.generate(prompt_batch, sampling_params=sampling_params)
        maybe_cuda_sync()
        total_output_tokens += sum(len(o.outputs[0].token_ids) for o in outputs)
        if verbose:
            print(f"[measure] batch={batch_idx}/{len(batches)} size={len(prompt_batch)}")

    wall_time_s = time.perf_counter() - t0
    after = snapshot_metrics(llm)

    num_drafts = metric_delta(after, before, SPEC_NUM_DRAFTS)
    num_draft_tokens = metric_delta(after, before, SPEC_NUM_DRAFT_TOKENS)
    num_accepted_tokens = metric_delta(after, before, SPEC_NUM_ACCEPTED)
    accepted_per_pos = vector_delta(after, before, SPEC_ACCEPTED_PER_POS)

    num_drafts_val = int(num_drafts) if num_drafts is not None else 0
    num_draft_tokens_val = int(num_draft_tokens) if num_draft_tokens is not None else 0
    num_accepted_val = int(num_accepted_tokens) if num_accepted_tokens is not None else 0

    avg_acceptance_rate = (
        num_accepted_val / num_draft_tokens_val if num_draft_tokens_val > 0 else 0.0
    )
    avg_acceptance_length = (
        1.0 + (num_accepted_val / num_drafts_val) if num_drafts_val > 0 else 1.0
    )
    acceptance_rate_per_pos = [
        (v / num_drafts_val) if num_drafts_val > 0 else 0.0
        for v in accepted_per_pos
    ]

    result: dict[str, Any] = {
        "num_prompts": len(prompts),
        "batch_size": batch_size,
        "wall_time_s": wall_time_s,
        "total_output_tokens": total_output_tokens,
        "num_drafts": num_drafts_val,
        "num_draft_tokens": num_draft_tokens_val,
        "num_accepted_tokens": num_accepted_val,
        "avg_acceptance_rate": avg_acceptance_rate,
        "avg_acceptance_length": avg_acceptance_length,
        "acceptance_rate_per_pos": acceptance_rate_per_pos,
    }

    if profile_time:
        draft_time_s = metric_delta(after, before, SPEC_DRAFT_TIME)
        verification_time_s = metric_delta(after, before, SPEC_VERIFICATION_TIME)
        draft_time_s_val = float(draft_time_s) if draft_time_s is not None else None
        verification_time_s_val = (
            float(verification_time_s) if verification_time_s is not None else None
        )
        result["draft_time_s"] = draft_time_s_val
        result["verification_time_s"] = verification_time_s_val
        if num_drafts_val > 0:
            result["avg_draft_time_s"] = (
                draft_time_s_val / num_drafts_val if draft_time_s_val is not None else None
            )
            result["avg_verification_time_s"] = (
                verification_time_s_val / num_drafts_val
                if verification_time_s_val is not None
                else None
            )
        else:
            result["avg_draft_time_s"] = None
            result["avg_verification_time_s"] = None

        draft_verif_checks = metric_delta(after, before, SPEC_DRAFT_VERIF_CHECKS)
        draft_verif_mismatches = metric_delta(after, before, SPEC_DRAFT_VERIF_MISMATCHES)
        checks_val = int(draft_verif_checks) if draft_verif_checks is not None else None
        mismatches_val = (
            int(draft_verif_mismatches) if draft_verif_mismatches is not None else None
        )
        result["draft_verification_checks"] = checks_val
        result["draft_verification_mismatches"] = mismatches_val
        result["draft_verification_match_ok"] = (
            (mismatches_val == 0 and checks_val is not None and checks_val > 0)
            if (checks_val is not None and mismatches_val is not None)
            else None
        )

    return result


def free_llm(llm: Any) -> None:
    try:
        del llm
    except Exception:
        pass
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def save_results(
    all_rows: list[dict[str, Any]],
    results_csv: str,
    results_jsonl: str,
) -> None:
    if not all_rows:
        return
    fieldnames = list(all_rows[0].keys())
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            r = dict(row)
            if "acceptance_rate_per_pos" in r and isinstance(
                r["acceptance_rate_per_pos"], list
            ):
                r["acceptance_rate_per_pos"] = json.dumps(r["acceptance_rate_per_pos"])
            writer.writerow(r)

    with open(results_jsonl, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[save] CSV: {results_csv}, JSONL: {results_jsonl}")


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)

    if args.profile_time:
        os.environ["VLLM_SPEC_PROFILE_TIME"] = "1"
        os.environ["VLLM_SPEC_VERIFY_DRAFT_MATCH"] = "1"
        print(
            "[profile_time] VLLM_SPEC_PROFILE_TIME=1, VLLM_SPEC_VERIFY_DRAFT_MATCH=1 "
            "(draft/verification timing and draft–verification match sampling enabled)"
        )
    else:
        os.environ["VLLM_SPEC_PROFILE_TIME"] = "0"
        os.environ["VLLM_SPEC_VERIFY_DRAFT_MATCH"] = "0"

    batch_sizes = [int(x) for x in parse_csv_list(args.batch_sizes)]
    target_models = parse_csv_list(args.target_models)
    dataset_keys = parse_csv_list(args.datasets)
    tp_map = parse_tp_map(args.tp_map)

    print("[args] draft_model =", args.draft_model)
    print("[args] target_models =", target_models)
    print("[args] tp_map =", tp_map)
    print("[args] profile_time =", args.profile_time)

    missing_tp = [m for m in target_models if m not in tp_map]
    if missing_tp:
        raise ValueError(f"Missing --tp-map for: {missing_tp}")

    dataset_prompts = {key: get_dataset_prompts(key, args) for key in dataset_keys}

    from vllm import LLM

    all_rows: list[dict[str, Any]] = []
    max_num_seqs = max(batch_sizes)

    for target_model in target_models:
        tp = tp_map[target_model]
        speculative_config = {
            "method": "draft_model",
            "model": args.draft_model,
            "num_speculative_tokens": args.num_spec_tokens,
            "max_model_len": args.max_model_len,
            "enforce_eager": args.enforce_eager,
        }

        print("=" * 80)
        print(f"[model-pair] draft={args.draft_model} | target={target_model} | tp={tp}")

        llm = LLM(
            model=target_model,
            tensor_parallel_size=tp,
            trust_remote_code=args.trust_remote_code,
            gpu_memory_utilization=args.gpu_memory_utilization,
            speculative_config=speculative_config,
            max_model_len=args.max_model_len,
            dtype=args.dtype,
            seed=args.seed,
            disable_log_stats=args.disable_log_stats,
            enforce_eager=args.enforce_eager,
            enable_chunked_prefill=args.enable_chunked_prefill,
            disable_custom_all_reduce=args.disable_custom_all_reduce,
            max_num_seqs=max_num_seqs,
        )
        tokenizer = llm.get_tokenizer()

        for dataset_key in dataset_keys:
            prompts = dataset_prompts[dataset_key]
            max_new_tokens = get_dataset_max_new_tokens(dataset_key, args)

            for batch_size in batch_sizes:
                print("-" * 80)
                print(
                    f"[run] dataset={dataset_key} target={target_model} "
                    f"batch_size={batch_size} num_prompts={len(prompts)} "
                    f"max_new_tokens={max_new_tokens}"
                )

                run_warmup(
                    llm=llm,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    batch_size=batch_size,
                    warmup_iters=args.warmup_iters,
                    warmup_max_tokens=args.warmup_max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                )

                metrics = measure_dataset(
                    llm=llm,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    batch_size=batch_size,
                    max_new_tokens=max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    profile_time=args.profile_time,
                    verbose=args.verbose,
                )

                row = {
                    "draft_model": args.draft_model,
                    "target_model": target_model,
                    "tensor_parallel_size": tp,
                    "dataset": dataset_key,
                    "batch_size": batch_size,
                    "num_speculative_tokens": args.num_spec_tokens,
                    "num_prompts": metrics["num_prompts"],
                    "wall_time_s": metrics["wall_time_s"],
                    "total_output_tokens": metrics["total_output_tokens"],
                    "num_drafts": metrics["num_drafts"],
                    "num_draft_tokens": metrics["num_draft_tokens"],
                    "num_accepted_tokens": metrics["num_accepted_tokens"],
                    "avg_acceptance_rate": metrics["avg_acceptance_rate"],
                    "avg_acceptance_length": metrics["avg_acceptance_length"],
                    "acceptance_rate_per_pos": metrics["acceptance_rate_per_pos"],
                }
                if args.profile_time:
                    row["draft_time_s"] = metrics.get("draft_time_s")
                    row["verification_time_s"] = metrics.get("verification_time_s")
                    row["avg_draft_time_s"] = metrics.get("avg_draft_time_s")
                    row["avg_verification_time_s"] = metrics.get("avg_verification_time_s")
                    row["draft_verification_checks"] = metrics.get("draft_verification_checks")
                    row["draft_verification_mismatches"] = metrics.get(
                        "draft_verification_mismatches"
                    )
                    row["draft_verification_match_ok"] = metrics.get(
                        "draft_verification_match_ok"
                    )
                else:
                    row["draft_time_s"] = None
                    row["verification_time_s"] = None
                    row["avg_draft_time_s"] = None
                    row["avg_verification_time_s"] = None
                    row["draft_verification_checks"] = None
                    row["draft_verification_mismatches"] = None
                    row["draft_verification_match_ok"] = None

                all_rows.append(row)

                print(
                    f"[result] dataset={dataset_key} target={target_model} "
                    f"batch_size={batch_size} wall_time_s={row['wall_time_s']:.4f} "
                    f"avg_acceptance_rate={row['avg_acceptance_rate']:.4f} "
                    f"avg_acceptance_length={row['avg_acceptance_length']:.4f}"
                )
                if args.profile_time:
                    ad = row.get("avg_draft_time_s")
                    av = row.get("avg_verification_time_s")
                    if ad is not None and av is not None:
                        print(
                            f"         avg_draft_time_s={ad:.6f} "
                            f"avg_verification_time_s={av:.6f}"
                        )
                    elif row.get("draft_time_s") is None and row.get("verification_time_s") is None:
                        print(
                            "[warning] draft/verification times are null. "
                            "Ensure --profile-time was passed and not --disable-log-stats."
                        )
                    checks = row.get("draft_verification_checks")
                    mismatches = row.get("draft_verification_mismatches")
                    match_ok = row.get("draft_verification_match_ok")
                    if checks is not None:
                        print(
                            f"         draft_verification_checks={checks} "
                            f"draft_verification_mismatches={mismatches} "
                            f"draft_verification_match_ok={match_ok}"
                        )

        save_results(all_rows, args.results_csv, args.results_jsonl)
        free_llm(llm)

    print("=" * 80)
    print(f"Results: {args.results_csv}, {args.results_jsonl}")
