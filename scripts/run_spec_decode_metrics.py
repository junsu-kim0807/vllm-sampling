#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Run vanilla speculative decoding (draft_model) on AIME 2025 and CodeElo.

This version keeps the original CLI behavior, but also saves per-pair results under:

    <results_root>/<sanitized_draft>__TO__<sanitized_target>/

Each pair directory contains:
  - metrics.csv
  - metrics.jsonl
  - pair_info.json
  - responses.jsonl (unless --no-responses): one JSON object per prompt (prompt_index, response, num_output_tokens).

If multiple targets are provided, an aggregate CSV/JSONL is also written under
results_root using --results-csv / --results-jsonl.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

from vllm.v1.metrics.reader import Counter, Gauge, Vector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run speculative decoding on AIME 2025 and CodeElo. "
            "With --profile-time, report average draft/verification time."
        )
    )
    p.add_argument("--draft-model", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument(
        "--method",
        type=str,
        default="speculative",
        choices=["speculative", "ar", "eagle3"],
        help="Evaluation method: 'ar' (no speculative decoding), 'speculative' (draft_model), 'eagle3' (EAGLE3 drafter).",
    )
    p.add_argument(
        "--eagle-model",
        type=str,
        default=None,
        help="When --method=eagle3: the speculative draft model repo id.",
    )
    p.add_argument(
        "--eagle-draft-tp",
        type=int,
        default=None,
        help="When --method=eagle3: draft_tensor_parallel_size override (default: same as target TP).",
    )
    p.add_argument(
        "--target-models",
        type=str,
        default="Qwen/Qwen3-8B,Qwen/Qwen3-30B-A3B",
        help="Comma-separated target model list.",
    )
    p.add_argument(
        "--tp-map",
        type=str,
        default="Qwen/Qwen3-8B=1,Qwen/Qwen3-30B-A3B=2",
        help="Comma-separated target_model=tp map.",
    )
    p.add_argument(
        "--datasets",
        type=str,
        default="aime25,codeelo",
        help="Comma-separated keys: aime25, codeelo, gov_report, qmsum (LongBench).",
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
        "--gov-report-max-new-tokens",
        type=int,
        default=512,
        help="Max new tokens for LongBench gov_report (summarization).",
    )
    p.add_argument(
        "--qmsum-max-new-tokens",
        type=int,
        default=512,
        help="Max new tokens for LongBench qmsum (summarization).",
    )
    p.add_argument(
        "--max-samples-gov-report",
        type=int,
        default=None,
        help="Optional cap for LongBench gov_report.",
    )
    p.add_argument(
        "--max-samples-qmsum",
        type=int,
        default=None,
        help="Optional cap for LongBench qmsum.",
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1)
    p.add_argument("--warmup-iters", type=int, default=2)
    p.add_argument("--warmup-max-tokens", type=int, default=32)
    p.add_argument(
        "--profile-time",
        action="store_true",
        help="Set VLLM_SPEC_PROFILE_TIME=1 and report average draft/verification time.",
    )
    p.add_argument(
        "--results-root",
        type=str,
        default="results/profile",
        help="Root directory for aggregate and per-pair outputs.",
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
    p.add_argument(
        "--responses-jsonl",
        type=str,
        default=None,
        help="Per-line JSON with only: prompt_index, response, num_output_tokens. Default: <pair_dir>/responses.jsonl.",
    )
    p.add_argument(
        "--no-responses",
        action="store_true",
        help="Do not write responses JSONL.",
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
LONGBENCH_REPO = "THUDM/LongBench"


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


def _load_longbench(cfg: str, max_samples: int | None) -> list[dict[str, Any]]:
    """Load a LongBench subset (e.g. gov_report, qmsum).

    - If LONGBENCH_DATA_DIR is set, load from that directory (after running
      scripts/download_longbench.py).
    - Else try load_dataset; if that fails (dataset scripts disabled), fall
      back to data.zip and find any file whose path contains the subset name
      and ends with test.jsonl / test.json or <cfg>.jsonl / <cfg>.json.
    """
    # 1) Prefer local dir from download_longbench.py
    data_dir = os.environ.get("LONGBENCH_DATA_DIR")
    if data_dir:
        data_path = Path(data_dir)
        if data_path.is_dir():
            # Look for data/<cfg>.jsonl, data/<cfg>/test.jsonl; or LongBench/data/... when zip has that root
            candidates = [
                data_path / "data" / f"{cfg}.jsonl",
                data_path / "data" / f"{cfg}.json",
                data_path / "data" / cfg / "test.jsonl",
                data_path / "data" / cfg / "test.json",
                data_path / "LongBench" / "data" / f"{cfg}.jsonl",
                data_path / "LongBench" / "data" / f"{cfg}.json",
                data_path / cfg / "test.jsonl",
                data_path / cfg / "test.json",
            ]
            for path in candidates:
                if path.is_file():
                    rows = []
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            rows.append(json.loads(line))
                    if max_samples is not None:
                        rows = rows[:max_samples]
                    return rows
            raise FileNotFoundError(
                f"LONGBENCH_DATA_DIR={data_dir} set but no file found for {cfg}. "
                f"Tried: {[str(c) for c in candidates]}. "
                "Run: python scripts/download_longbench.py --out-dir <dir>"
            )

    from datasets import load_dataset

    try:
        ds = load_dataset(LONGBENCH_REPO, cfg, split="test")
        rows = list(ds)
    except RuntimeError as e:
        msg = str(e)
        if "no longer supported" in msg or "Dataset scripts" in msg:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as ie:
                raise RuntimeError(
                    "LongBench script loading is disabled. Either set LONGBENCH_DATA_DIR "
                    "to a dir created by scripts/download_longbench.py, or install "
                    "huggingface_hub for fallback: pip install huggingface_hub"
                ) from ie

            zip_path = hf_hub_download(
                repo_id=LONGBENCH_REPO,
                filename="data.zip",
                repo_type="dataset",
            )

            import zipfile

            rows = []
            with zipfile.ZipFile(zip_path, "r") as zf:
                namelist = zf.namelist()
                # Match any member that looks like this subset's test data
                def _matches(name: str) -> bool:
                    if cfg not in name:
                        return False
                    lower = name.lower()
                    if lower.endswith(".jsonl") or lower.endswith(".json"):
                        # data/gov_report.jsonl, data/gov_report/test.jsonl, data/test_gov_report.jsonl, etc.
                        return "test" in lower or name.endswith(f"{cfg}.jsonl") or name.endswith(f"{cfg}.json")
                    return False

                picked = next((n for n in namelist if _matches(n)), None)
                if picked is None:
                    # Fallback: any path containing cfg and ending in .jsonl
                    picked = next(
                        (n for n in namelist if cfg in n and n.endswith(".jsonl")),
                        None,
                    )
                if picked is None:
                    raise RuntimeError(
                        f"Could not locate LongBench {cfg} inside data.zip. "
                        "Run: python scripts/download_longbench.py --out-dir DIR "
                        "then set LONGBENCH_DATA_DIR=DIR"
                    )

                with zf.open(picked, "r") as f:
                    for raw in f:
                        line = raw.decode("utf-8").strip()
                        if not line:
                            continue
                        rows.append(json.loads(line))
        else:
            raise

    if max_samples is not None:
        rows = rows[:max_samples]
    return rows


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

    def _build_longbench_prompt_from_example(ex: dict[str, Any]) -> str:
        # LongBench JSON schemas vary slightly across forks/packaging.
        # We prefer instruction-like fields first, then fall back to the
        # main source text fields.
        candidate_keys = [
            "input",
            "prompt",
            "question",
            "query",
            "instruction",
            "context",
            "article",
            "document",
            "passage",
            "text",
            "source",
        ]
        for k in candidate_keys:
            if k in ex and ex[k] is not None:
                s = str(ex[k]).strip()
                if s:
                    return s
        return ""

    if dataset_key in ("gov_report", "longbench_gov_report"):
        cfg = "gov_report"
        max_samples = args.max_samples_gov_report
        rows = _load_longbench(cfg, max_samples)
        if not rows:
            raise RuntimeError(
                f"LongBench {cfg}: loaded 0 rows. "
                f"Check LONGBENCH_DATA_DIR or the downloaded data.zip."
            )
        prompts = [_build_longbench_prompt_from_example(ex) for ex in rows]
        prompts = [p for p in prompts if p]
        if not prompts:
            first = rows[0]
            keys = list(first.keys())
            raise RuntimeError(
                f"LongBench {cfg}: all prompt fields were empty after extraction. "
                f"First-row keys={keys}. "
                f"Candidate prompt keys are input/prompt/question/query/instruction/context/article/document/passsage/text/source. "
                f"Set LONGBENCH_DATA_DIR to the extracted LongBench directory and re-run "
                f"(or adjust _build_longbench_prompt_from_example for this schema)."
            )
        print(
            f"[dataset] {dataset_key}: repo={LONGBENCH_REPO}/{cfg}, "
            f"split=test, samples={len(prompts)}"
        )
        return prompts

    if dataset_key in ("qmsum", "longbench_qmsum"):
        cfg = "qmsum"
        max_samples = args.max_samples_qmsum
        rows = _load_longbench(cfg, max_samples)
        if not rows:
            raise RuntimeError(
                f"LongBench {cfg}: loaded 0 rows. "
                f"Check LONGBENCH_DATA_DIR or the downloaded data.zip."
            )
        prompts = [_build_longbench_prompt_from_example(ex) for ex in rows]
        prompts = [p for p in prompts if p]
        if not prompts:
            first = rows[0]
            keys = list(first.keys())
            raise RuntimeError(
                f"LongBench {cfg}: all prompt fields were empty after extraction. "
                f"First-row keys={keys}. "
                f"Candidate prompt keys are input/prompt/question/query/instruction/context/article/document/passsage/text/source. "
                f"Set LONGBENCH_DATA_DIR to the extracted LongBench directory and re-run "
                f"(or adjust _build_longbench_prompt_from_example for this schema)."
            )
        print(
            f"[dataset] {dataset_key}: repo={LONGBENCH_REPO}/{cfg}, "
            f"split=test, samples={len(prompts)}"
        )
        return prompts

    raise ValueError(f"Unsupported dataset key: {dataset_key}")


def get_dataset_max_new_tokens(dataset_key: str, args: argparse.Namespace) -> int:
    if dataset_key == "aime25":
        return args.aime_max_new_tokens
    if dataset_key == "codeelo":
        return args.codeelo_max_new_tokens
    if dataset_key in ("gov_report", "longbench_gov_report"):
        return args.gov_report_max_new_tokens
    if dataset_key in ("qmsum", "longbench_qmsum"):
        return args.qmsum_max_new_tokens
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
SPEC_REJECT_SAMPLE_TIME = "vllm:spec_decode_reject_sample_time_seconds_total"


def chunked(prompts: list[str], batch_size: int):
    for i in range(0, len(prompts), batch_size):
        yield prompts[i : i + batch_size]


def expand_prompts_for_batch(
    prompts: list[str], batch_size: int
) -> tuple[list[str], int]:
    """If dataset prompt count is smaller than 2 * batch_size, tile prompts
    until len == 2 * batch_size.

    Returns (prompts_to_run, num_prompts_dataset) where num_prompts_dataset is
    the original row count before tiling.
    """
    n = len(prompts)
    if n == 0 or batch_size < 1:
        return list(prompts), n
    if n >= 2 * batch_size:
        return list(prompts), n
    target = 2 * batch_size
    tiled = [prompts[i % n] for i in range(target)]
    return tiled, n


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
    print(
        f"[warmup] iters={warmup_iters}, batch_size={batch_size}, "
        f"max_tokens={warmup_max_tokens}"
    )
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
    *,
    draft_model: str = "",
    target_model: str = "",
    dataset: str = "",
    collect_responses: bool = False,
    num_prompts_source: int | None = None,
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
    response_records: list[dict[str, Any]] = []
    prompt_offset = 0

    maybe_cuda_sync()
    t0 = time.perf_counter()

    for batch_idx, prompt_batch in enumerate(batches, start=1):
        maybe_cuda_sync()
        outputs = llm.generate(prompt_batch, sampling_params=sampling_params)
        maybe_cuda_sync()
        total_output_tokens += sum(len(o.outputs[0].token_ids) for o in outputs)
        if collect_responses:
            for local_i, req_out in enumerate(outputs):
                comp = req_out.outputs[0]
                gidx = prompt_offset + local_i
                response_records.append(
                    {
                        "prompt_index": gidx,
                        "response": comp.text,
                        "num_output_tokens": len(comp.token_ids),
                    }
                )
            prompt_offset += len(prompt_batch)
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

        reject_sample_time = metric_delta(after, before, SPEC_REJECT_SAMPLE_TIME)
        result["reject_sample_time_s"] = (
            float(reject_sample_time) if reject_sample_time is not None else None
        )
        if num_drafts_val > 0 and result["reject_sample_time_s"] is not None:
            result["avg_reject_sample_time_s"] = (
                result["reject_sample_time_s"] / num_drafts_val
            )
        else:
            result["avg_reject_sample_time_s"] = None

    if collect_responses:
        result["response_records"] = response_records
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


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def save_results(
    all_rows: list[dict[str, Any]],
    results_csv: str,
    results_jsonl: str,
) -> None:
    if not all_rows:
        return

    ensure_parent_dir(results_csv)
    ensure_parent_dir(results_jsonl)

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


def save_responses_jsonl(records: list[dict[str, Any]], path: str) -> None:
    if not records or not path:
        return
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[save] responses JSONL: {path} ({len(records)} lines)")


def sanitize_model_name(model_name: str) -> str:
    cleaned = model_name.strip().replace("/", "__")
    cleaned = re.sub(r"[^A-Za-z0-9._+@=,]+", "_", cleaned)
    cleaned = cleaned.strip("._")
    return cleaned


def get_pair_output_dir(results_root: str, draft_model: str, target_model: str) -> str:
    pair_name = f"{sanitize_model_name(draft_model)}__TO__{sanitize_model_name(target_model)}"
    return os.path.join(results_root, pair_name)


def write_pair_info(
    pair_dir: str,
    args: argparse.Namespace,
    target_model: str,
    tensor_parallel_size: int,
    dataset_keys: list[str],
    batch_sizes: list[int],
) -> None:
    os.makedirs(pair_dir, exist_ok=True)
    payload = {
        "draft_model": args.draft_model,
        "target_model": target_model,
        "tensor_parallel_size": tensor_parallel_size,
        "datasets": dataset_keys,
        "batch_sizes": batch_sizes,
        "num_speculative_tokens": args.num_spec_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "dtype": args.dtype,
        "seed": args.seed,
        "profile_time": args.profile_time,
    }
    path = os.path.join(pair_dir, "pair_info.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[save] pair metadata: {path}")


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
    print("[args] results_root =", args.results_root)

    if args.no_responses:
        responses_jsonl_path = None
    elif args.responses_jsonl:
        responses_jsonl_path = args.responses_jsonl
    else:
        responses_jsonl_path = "responses.jsonl"
    if responses_jsonl_path and not args.no_responses:
        print("[args] responses_jsonl = (per-pair)", responses_jsonl_path)

    missing_tp = [m for m in target_models if m not in tp_map]
    if missing_tp:
        raise ValueError(f"Missing --tp-map for: {missing_tp}")

    os.makedirs(args.results_root, exist_ok=True)
    dataset_prompts = {key: get_dataset_prompts(key, args) for key in dataset_keys}

    from vllm import LLM

    all_rows: list[dict[str, Any]] = []
    max_num_seqs = max(batch_sizes)

    for target_model in target_models:
        tp = tp_map[target_model]
        if args.method == "ar":
            speculative_config = None
        elif args.method == "eagle3":
            if not args.eagle_model:
                raise SystemExit("--method=eagle3 requires --eagle-model")
            draft_tp = args.eagle_draft_tp if args.eagle_draft_tp else tp
            speculative_config = {
                "method": "eagle3",
                "model": args.eagle_model,
                "draft_tensor_parallel_size": draft_tp,
                "num_speculative_tokens": args.num_spec_tokens,
            }
        else:
            # Speculative decoding with a draft model.
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
        target_rows: list[dict[str, Any]] = []
        pair_response_records: list[dict[str, Any]] = []

        for dataset_key in dataset_keys:
            prompts = dataset_prompts[dataset_key]
            max_new_tokens = get_dataset_max_new_tokens(dataset_key, args)

            for batch_size in batch_sizes:
                prompts_run, n_dataset = expand_prompts_for_batch(prompts, batch_size)
                if len(prompts_run) > n_dataset:
                    print(
                        f"[dataset] tiled {n_dataset} -> {len(prompts_run)} prompts "
                        f"(batch_size={batch_size}, need >= 2*batch for small sets)"
                    )
                print("-" * 80)
                print(
                    f"[run] dataset={dataset_key} target={target_model} "
                    f"batch_size={batch_size} num_prompts_dataset={n_dataset} "
                    f"num_prompts_eval={len(prompts_run)} max_new_tokens={max_new_tokens}"
                )

                run_warmup(
                    llm=llm,
                    tokenizer=tokenizer,
                    prompts=prompts_run,
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
                    prompts=prompts_run,
                    batch_size=batch_size,
                    max_new_tokens=max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    profile_time=args.profile_time,
                    verbose=args.verbose,
                    draft_model=args.draft_model,
                    target_model=target_model,
                    dataset=dataset_key,
                    collect_responses=bool(responses_jsonl_path and not args.no_responses),
                    num_prompts_source=n_dataset,
                )
                if responses_jsonl_path and metrics.get("response_records"):
                    pair_response_records.extend(metrics["response_records"])

                row = {
                    "draft_model": args.draft_model,
                    "target_model": target_model,
                    "tensor_parallel_size": tp,
                    "dataset": dataset_key,
                    "batch_size": batch_size,
                    "num_speculative_tokens": args.num_spec_tokens,
                    "num_prompts_dataset": n_dataset,
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
                    row["reject_sample_time_s"] = metrics.get("reject_sample_time_s")
                    row["avg_reject_sample_time_s"] = metrics.get(
                        "avg_reject_sample_time_s"
                    )
                else:
                    row["draft_time_s"] = None
                    row["verification_time_s"] = None
                    row["avg_draft_time_s"] = None
                    row["avg_verification_time_s"] = None
                    row["reject_sample_time_s"] = None
                    row["avg_reject_sample_time_s"] = None

                target_rows.append(row)
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
                    rs = row.get("reject_sample_time_s")
                    ars = row.get("avg_reject_sample_time_s")
                    if rs is not None:
                        ars_str = f"{ars:.6f}" if ars is not None else "N/A"
                        print(
                            f"         reject_sample_time_s={rs:.6f} "
                            f"avg_reject_sample_time_s={ars_str}"
                        )

        pair_dir = get_pair_output_dir(args.results_root, args.draft_model, target_model)
        pair_csv = os.path.join(pair_dir, "metrics.csv")
        pair_jsonl = os.path.join(pair_dir, "metrics.jsonl")
        save_results(target_rows, pair_csv, pair_jsonl)
        write_pair_info(
            pair_dir=pair_dir,
            args=args,
            target_model=target_model,
            tensor_parallel_size=tp,
            dataset_keys=dataset_keys,
            batch_sizes=batch_sizes,
        )
        if responses_jsonl_path and pair_response_records:
            save_responses_jsonl(
                pair_response_records,
                os.path.join(pair_dir, responses_jsonl_path),
            )
        free_llm(llm)

    aggregate_csv = (
        args.results_csv
        if os.path.isabs(args.results_csv)
        else os.path.join(args.results_root, args.results_csv)
    )
    aggregate_jsonl = (
        args.results_jsonl
        if os.path.isabs(args.results_jsonl)
        else os.path.join(args.results_root, args.results_jsonl)
    )
    save_results(all_rows, aggregate_csv, aggregate_jsonl)

    print("=" * 80)
    print(f"Aggregate results: {aggregate_csv}, {aggregate_jsonl}")
    print(f"Per-pair results root: {args.results_root}")
    if responses_jsonl_path:
        print("(Per-pair responses: <pair_dir>/responses.jsonl)")
