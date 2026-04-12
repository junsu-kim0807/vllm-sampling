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
  - responses.jsonl (unless --no-responses): one JSON object per request.
    For multi-turn datasets it also includes sample_id and turn_index.

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

def is_speculative_method(method: str) -> bool:
    return method in (
        "speculative",
        "eagle3",
        "magicdec",
        "adaptive_spechive",
        "pivot",
        "tetris",
    )

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
        choices=["speculative", "ar", "eagle3", "magicdec", "adaptive_spechive", "pivot", "tetris"],
        help=(
            "Evaluation method: 'ar' (no speculative decoding), "
            "'speculative' (draft_model), 'eagle3' (EAGLE3 drafter), "
            "'magicdec' (draft_model + magicdec streaming KV view), "
            "'adaptive_spechive' (draft + intermediate staged proposer), "
            "'pivot' (intermediate pivot token + draft tail), "
            "'tetris' (TETRIS optimal draft token selection, ACL 2025)."
        ),
    )
    p.add_argument(
        "--tetris-extra-proposals",
        type=int,
        default=2,
        help=(
            "When --method=tetris: extra draft tokens beyond base_k. "
            "base_k = num_spec_tokens (what you'd use in vanilla). "
            "Drafter automatically generates K = base_k + extra tokens; "
            "capacity = base_k * batch_size. "
            "Paper experiments used 1, 2, or 3 (default: 2)."
        ),
    )
    p.add_argument(
        "--tetris-turn-on-batch-size",
        type=int,
        default=None,
        help=(
            "When --method=tetris: minimum batch size before TETRIS activates; "
            "smaller batches fall back to standard spec-decode (default: always on)."
        ),
    )
    p.add_argument(
        "--intermediate-model",
        type=str,
        default=None,
        help=(
            "When --method=adaptive_spechive: intermediate verifier model id."
        ),
    )
    p.add_argument(
        "--topk_selection",
        type=int,
        default=5,
        choices=[2, 5],
        help=(
            "When --method=pivot: first-token top-k expansion width "
            "(default: 5; allowed: 2 or 5)."
        ),
    )
    p.add_argument(
        "--expansion_pct",
        type=float,
        default=0.2,
        help=(
            "When --method=pivot: fraction of low-confidence requests to expand "
            "(default: 0.2)."
        ),
    )
    p.add_argument(
        "--spechive",
        action="store_true",
        help=(
            "When --method=pivot: enable staged D=>I rounds before final D=>T "
            "verification."
        ),
    )
    p.add_argument(
        "--adaptive-spechive-mode",
        type=str,
        default="hierarchical_verification",
        choices=["draft_target", "inter_verification", "hierarchical_verification"],
        help="When --method=adaptive_spechive: adaptive spechive mode.",
    )
    p.add_argument(
        "--round",
        type=int,
        default=1,
        help=(
            "When --method=adaptive_spechive: number of hierarchical verification rounds."
        ),
    )
    p.add_argument(
        "--magicdec-method",
        type=str,
        default="streaming",
        choices=["streaming"],
        help="When --method=magicdec: magicdec method (default: streaming).",
    )
    p.add_argument(
        "--magicdec-kv-budget",
        type=int,
        default=256,
        help="When --method=magicdec: visible KV token budget (default: 256).",
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
        help=(
            "Comma-separated keys: aime25, codeelo, gov_report, qmsum "
            "(LongBench), spec_bench, alpaca, gsm8k, mt_bench, qa, humaneval, sum."
        ),
    )
    p.add_argument(
        "--repo-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1]),
        help=(
            "Repository root containing data/<dataset>/question.jsonl for "
            "Spec-Bench split datasets."
        ),
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
    p.add_argument(
        "--spec-bench-jsonl",
        type=str,
        default=None,
        help=(
            "Path to Spec-Bench question.jsonl. Required when using "
            "--datasets spec_bench."
        ),
    )
    p.add_argument(
        "--spec-bench-category",
        type=str,
        default=None,
        help=(
            "Optional Spec-Bench category filter (e.g. alpaca, mt_bench). "
            "Comma-separated values are allowed."
        ),
    )
    p.add_argument(
        "--spec-bench-max-new-tokens",
        type=int,
        default=256,
        help="Max new tokens for Spec-Bench.",
    )
    p.add_argument(
        "--max-samples-spec-bench",
        type=int,
        default=None,
        help="Optional cap for Spec-Bench samples after category filtering.",
    )
    p.add_argument(
        "--mt-bench-dataset-path",
        type=str,
        default="philschmid/mt-bench",
        help=(
            "Hugging Face dataset path or local JSONL path for MT-Bench "
            "(must contain a turns field)."
        ),
    )
    p.add_argument(
        "--mt-bench-max-new-tokens",
        type=int,
        default=256,
        help="Max new tokens for MT-Bench.",
    )
    p.add_argument(
        "--max-samples-mt-bench",
        type=int,
        default=None,
        help="Optional cap for MT-Bench samples.",
    )
    p.add_argument("--alpaca-max-new-tokens", type=int, default=256)
    p.add_argument("--gsm8k-max-new-tokens", type=int, default=256)
    p.add_argument("--qa-max-new-tokens", type=int, default=256)
    p.add_argument("--humaneval-max-new-tokens", type=int, default=512)
    p.add_argument("--sum-max-new-tokens", type=int, default=512)
    p.add_argument("--max-samples-alpaca", type=int, default=None)
    p.add_argument("--max-samples-gsm8k", type=int, default=None)
    p.add_argument("--max-samples-qa", type=int, default=None)
    p.add_argument("--max-samples-humaneval", type=int, default=None)
    p.add_argument("--max-samples-sum", type=int, default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1)
    p.add_argument("--warmup-iters", type=int, default=10)
    p.add_argument("--warmup-max-tokens", type=int, default=1024)
    p.add_argument(
        "--profile-time",
        action="store_true",
        help=(
            "[Legacy] Alias for --spec-decode-profile-mode=stage_cost. "
            "Kept for backward compatibility."
        ),
    )
    p.add_argument(
        "--spec-decode-profile-mode",
        type=str,
        default="disabled",
        choices=["disabled", "stage_cost", "shape_memory", "kernel_breakdown", "all"],
        help=(
            "Unified speculative-decoding profiler mode. "
            "'stage_cost' enables per-stage wall-time collection "
            "(draft_forward, intermediate_verify, target_verify, "
            "expand_collapse, reject_sample, bookkeeping). "
            "'shape_memory' adds memory/shape snapshots. "
            "'kernel_breakdown' enables sampled deep kernel profiling. "
            "'all' enables everything. Default: disabled."
        ),
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
        help=(
            "Per-line JSONL responses. Single-turn records contain prompt_index; "
            "multi-turn records also include sample_id and turn_index. "
            "Default: <pair_dir>/responses.jsonl."
        ),
    )
    p.add_argument(
        "--no-responses",
        action="store_true",
        help="Do not write responses JSONL.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Enable DIT runtime debug validation "
            "(sets VLLM_SPEC_DIT_DEBUG=1, VLLM_SPEC_DIT_DEBUG_SUMMARY=1)."
        ),
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
DEFAULT_MT_BENCH_REPO = "philschmid/mt-bench"
SPEC_BENCH_DATASET_KEYS = (
    "alpaca",
    "gsm8k",
    "mt_bench",
    "qa",
    "humaneval",
    "sum",
)


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"Invalid JSON at {path}:{line_no}: {e}"
                ) from e
            if not isinstance(obj, dict):
                raise RuntimeError(
                    f"Expected JSON object at {path}:{line_no}, got {type(obj)}"
                )
            rows.append(obj)
    return rows


def _normalize_turns(raw_turns: Any) -> list[str]:
    if not isinstance(raw_turns, list):
        return []
    turns = [str(x).strip() for x in raw_turns if str(x).strip()]
    return turns


def _resolve_repo_dataset_jsonl(repo_dir: str, dataset_key: str) -> Path:
    return Path(repo_dir) / "data" / dataset_key / "question.jsonl"


def _load_repo_dataset_conversations(
    repo_dir: str, dataset_key: str, max_samples: int | None
) -> list[list[str]]:
    jsonl_path = _resolve_repo_dataset_jsonl(repo_dir, dataset_key)
    if not jsonl_path.is_file():
        raise FileNotFoundError(
            f"Dataset file not found: {jsonl_path}. Expected REPO_DIR/data/{dataset_key}/question.jsonl"
        )
    rows = _load_jsonl_records(jsonl_path)
    conversations: list[list[str]] = []
    for row in rows:
        turns = _normalize_turns(row.get("turns"))
        if not turns:
            continue
        conversations.append(turns)
        if max_samples is not None and len(conversations) >= max_samples:
            break
    if not conversations:
        raise RuntimeError(
            f"{dataset_key}: 0 valid rows with non-empty 'turns' in {jsonl_path}"
        )
    print(
        f"[dataset] {dataset_key}: path={jsonl_path}, samples={len(conversations)}, mode=multi_turn"
    )
    return conversations


def _load_mt_like_conversations(
    dataset_path: str, max_samples: int | None
) -> list[list[str]]:
    rows: list[dict[str, Any]]
    local_jsonl = Path(dataset_path)
    if local_jsonl.is_file() and local_jsonl.suffix.lower() == ".jsonl":
        rows = _load_jsonl_records(local_jsonl)
        split_name = "jsonl"
    else:
        ds, split_name = load_dataset_split(dataset_path)
        rows = list(ds)

    conversations: list[list[str]] = []
    for row in rows:
        turns = _normalize_turns(row.get("turns"))
        if not turns:
            continue
        conversations.append(turns)
        if max_samples is not None and len(conversations) >= max_samples:
            break
    if not conversations:
        raise RuntimeError(
            f"MT-like dataset {dataset_path} split={split_name} produced 0 valid rows with turns."
        )
    return conversations


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
    - Else try load_dataset; if that fails (dataset scripts disabled, or
      partial local cache e.g. only another config like musique), fall back to
      Hub data.zip and find any file whose path contains the subset name and
      ends with test.jsonl / test.json or <cfg>.jsonl / <cfg>.json.
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

    def _longbench_zip_fallback() -> list[dict[str, Any]]:
        """Load subset from Hub data.zip (avoids broken/partial datasets cache)."""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as ie:
            raise RuntimeError(
                "LongBench could not be loaded from cache. Either set LONGBENCH_DATA_DIR "
                "to a dir created by scripts/download_longbench.py, or install "
                "huggingface_hub for data.zip fallback: pip install huggingface_hub"
            ) from ie

        zip_path = hf_hub_download(
            repo_id=LONGBENCH_REPO,
            filename="data.zip",
            repo_type="dataset",
        )

        import zipfile

        out: list[dict[str, Any]] = []
        with zipfile.ZipFile(zip_path, "r") as zf:
            namelist = zf.namelist()

            def _matches(name: str) -> bool:
                if cfg not in name:
                    return False
                lower = name.lower()
                if lower.endswith(".jsonl") or lower.endswith(".json"):
                    return (
                        "test" in lower
                        or name.endswith(f"{cfg}.jsonl")
                        or name.endswith(f"{cfg}.json")
                    )
                return False

            picked = next((n for n in namelist if _matches(n)), None)
            if picked is None:
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
                    out.append(json.loads(line))
        return out

    try:
        ds = load_dataset(LONGBENCH_REPO, cfg, split="test")
        rows = list(ds)
    except (RuntimeError, ValueError) as e:
        msg = str(e)
        # HF datasets: script disabled; or local cache only has other configs (e.g. musique)
        # and raises ValueError "Couldn't find cache ... for config 'qmsum'".
        use_zip_fallback = (
            "no longer supported" in msg
            or "Dataset scripts" in msg
            or "Couldn't find cache" in msg
            or "Available configs in the cache" in msg
        )
        if not use_zip_fallback:
            raise
        rows = _longbench_zip_fallback()

    if max_samples is not None:
        rows = rows[:max_samples]
    return rows


def get_dataset_requests(dataset_key: str, args: argparse.Namespace) -> dict[str, Any]:
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
        return {"mode": "single_turn", "prompts": prompts}

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
        return {"mode": "single_turn", "prompts": prompts}

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
        return {"mode": "single_turn", "prompts": prompts}

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
        return {"mode": "single_turn", "prompts": prompts}

    if dataset_key in SPEC_BENCH_DATASET_KEYS:
        max_samples_map = {
            "alpaca": args.max_samples_alpaca,
            "gsm8k": args.max_samples_gsm8k,
            "mt_bench": args.max_samples_mt_bench,
            "qa": args.max_samples_qa,
            "humaneval": args.max_samples_humaneval,
            "sum": args.max_samples_sum,
        }
        conversations = _load_repo_dataset_conversations(
            repo_dir=args.repo_dir,
            dataset_key=dataset_key,
            max_samples=max_samples_map[dataset_key],
        )
        return {"mode": "multi_turn", "conversations": conversations}

    if dataset_key == "spec_bench":
        if not args.spec_bench_jsonl:
            raise ValueError(
                "--spec-bench-jsonl is required when --datasets includes spec_bench."
            )
        jsonl_path = Path(args.spec_bench_jsonl)
        if not jsonl_path.is_file():
            raise FileNotFoundError(
                f"--spec-bench-jsonl not found: {args.spec_bench_jsonl}"
            )
        rows = _load_jsonl_records(jsonl_path)
        categories = set(parse_csv_list(args.spec_bench_category)) if args.spec_bench_category else None

        conversations: list[list[str]] = []
        for row in rows:
            if categories is not None:
                row_category = str(row.get("category", "")).strip()
                if row_category not in categories:
                    continue
            turns = _normalize_turns(row.get("turns"))
            if not turns:
                continue
            conversations.append(turns)
            if (
                args.max_samples_spec_bench is not None
                and len(conversations) >= args.max_samples_spec_bench
            ):
                break

        if not conversations:
            raise RuntimeError(
                "Spec-Bench produced 0 valid samples. Check --spec-bench-jsonl "
                "and --spec-bench-category filters."
            )
        cat_label = ",".join(sorted(categories)) if categories else "ALL"
        print(
            f"[dataset] {dataset_key}: path={jsonl_path}, "
            f"category={cat_label}, samples={len(conversations)}, mode=multi_turn"
        )
        return {"mode": "multi_turn", "conversations": conversations}

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
    if dataset_key == "spec_bench":
        return args.spec_bench_max_new_tokens
    if dataset_key == "alpaca":
        return args.alpaca_max_new_tokens
    if dataset_key == "gsm8k":
        return args.gsm8k_max_new_tokens
    if dataset_key == "mt_bench":
        return args.mt_bench_max_new_tokens
    if dataset_key == "qa":
        return args.qa_max_new_tokens
    if dataset_key == "humaneval":
        return args.humaneval_max_new_tokens
    if dataset_key == "sum":
        return args.sum_max_new_tokens
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

# Unified profiler stage-level metrics (cumulative ms, set as Gauge).
SPEC_DRAFT_FORWARD_MS = "vllm:spec_decode_draft_forward_ms_total"
SPEC_TARGET_VERIFY_MS = "vllm:spec_decode_target_verify_ms_total"
SPEC_INTERMEDIATE_VERIFY_MS = "vllm:spec_decode_intermediate_verify_ms_total"
SPEC_EXPAND_COLLAPSE_MS = "vllm:spec_decode_expand_collapse_ms_total"


def chunked(items: list[Any], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


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


def expand_conversations_for_batch(
    conversations: list[list[str]], batch_size: int
) -> tuple[list[list[str]], int]:
    n = len(conversations)
    if n == 0 or batch_size < 1:
        return list(conversations), n
    if n >= 2 * batch_size:
        return list(conversations), n
    target = 2 * batch_size
    tiled = [conversations[i % n] for i in range(target)]
    return tiled, n


def render_chat_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def apply_chat_template(tokenizer, prompts: list[str]) -> list[str]:
    out: list[str] = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = render_chat_prompt(tokenizer, messages)
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
    *,
    method: str,
    num_spec_tokens: int,
    profile_time: bool,
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

    speculative = is_speculative_method(method)

    # Speculative warmup에서는 draft/verify가 여러 번 돌 수 있게
    # 너무 짧은 warmup decode 길이를 피한다.
    effective_warmup_max_tokens = warmup_max_tokens
    if speculative:
        effective_warmup_max_tokens = max(
            warmup_max_tokens,
            max(8, num_spec_tokens * 4),
        )

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=effective_warmup_max_tokens,
    )

    print(
        f"[warmup] iters={warmup_iters}, batch_size={batch_size}, "
        f"max_tokens={effective_warmup_max_tokens}, speculative={speculative}"
    )

    # 기본 warmup은 수행하고,
    # speculative이면 실제 draft/verify path가 관측될 때까지 추가 warmup 허용.
    extra_budget = 4 if speculative else 0
    total_iters = 0

    saw_draft = False
    saw_verify = False

    while True:
        before = snapshot_metrics(llm) if speculative else {}

        maybe_cuda_sync()
        _ = llm.generate(warm, sampling_params=sampling_params)
        maybe_cuda_sync()

        after = snapshot_metrics(llm) if speculative else {}

        total_iters += 1

        if speculative:
            num_drafts = metric_delta(after, before, SPEC_NUM_DRAFTS)
            num_draft_tokens = metric_delta(after, before, SPEC_NUM_DRAFT_TOKENS)
            verification_time_s = metric_delta(after, before, SPEC_VERIFICATION_TIME)

            num_drafts_val = int(num_drafts) if num_drafts is not None else 0
            num_draft_tokens_val = (
                int(num_draft_tokens) if num_draft_tokens is not None else 0
            )
            verification_time_val = (
                float(verification_time_s) if verification_time_s is not None else None
            )

            if num_drafts_val > 0 or num_draft_tokens_val > 0:
                saw_draft = True

            # profiling이 켜져 있으면 verification_time 또는 unified target_verify
            # metric으로 확인. 꺼져 있으면 draft 발생 자체를 proxy로 본다.
            if profile_time:
                if verification_time_val is not None and verification_time_val > 0:
                    saw_verify = True
                else:
                    tv_delta = metric_delta(after, before, SPEC_TARGET_VERIFY_MS)
                    if tv_delta is not None and float(tv_delta) > 0:
                        saw_verify = True
            else:
                if saw_draft:
                    saw_verify = True

            print(
                f"[warmup] done {total_iters} "
                f"(drafts={num_drafts_val}, draft_tokens={num_draft_tokens_val}, "
                f"verify_time_s={verification_time_val})"
            )
        else:
            print(f"[warmup] done {total_iters}")

        base_done = total_iters >= warmup_iters
        spec_ready = (not speculative) or (saw_draft and saw_verify)

        if base_done and spec_ready:
            break

        if base_done and speculative and not spec_ready:
            if extra_budget <= 0:
                print(
                    "[warmup] warning: speculative warmup ended before both draft "
                    "and verifier were clearly observed."
                )
                break
            extra_budget -= 1


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

        _collect_unified_stage_metrics(result, after, before, num_drafts_val)

    if collect_responses:
        result["response_records"] = response_records
    return result


def _collect_unified_stage_metrics(
    result: dict[str, Any],
    after: dict[str, Any],
    before: dict[str, Any],
    num_drafts: int,
) -> None:
    """Collect unified profiler stage-level cost breakdown (ms) into result dict."""
    stage_metrics = {
        "draft_forward_ms": SPEC_DRAFT_FORWARD_MS,
        "target_verify_ms": SPEC_TARGET_VERIFY_MS,
        "intermediate_verify_ms": SPEC_INTERMEDIATE_VERIFY_MS,
        "expand_collapse_ms": SPEC_EXPAND_COLLAPSE_MS,
    }
    for field, metric_name in stage_metrics.items():
        delta = metric_delta(after, before, metric_name)
        val = float(delta) if delta is not None else None
        result[field] = val
        avg_field = f"avg_{field}"
        if num_drafts > 0 and val is not None:
            result[avg_field] = val / num_drafts
        else:
            result[avg_field] = None


def measure_dataset_multi_turn(
    llm: Any,
    tokenizer: Any,
    conversations: list[list[str]],
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    profile_time: bool,
    verbose: bool = False,
    *,
    collect_responses: bool = False,
) -> dict[str, Any]:
    from vllm import SamplingParams

    conv_batches = [
        list(batch) for batch in chunked(list(conversations), batch_size)
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
    num_turns_total = 0
    global_sample_offset = 0

    maybe_cuda_sync()
    t0 = time.perf_counter()

    for batch_idx, conv_batch in enumerate(conv_batches, start=1):
        # Keep per-sample chat history and append model-generated assistant
        # outputs turn by turn.
        histories: list[list[dict[str, str]]] = [[] for _ in conv_batch]
        max_turns = max(len(conv) for conv in conv_batch)

        for turn_idx in range(max_turns):
            active_indices: list[int] = []
            prompt_batch: list[str] = []

            for sample_idx, turns in enumerate(conv_batch):
                if turn_idx >= len(turns):
                    continue
                user_turn = turns[turn_idx]
                histories[sample_idx].append({"role": "user", "content": user_turn})
                prompt_batch.append(
                    render_chat_prompt(tokenizer, histories[sample_idx])
                )
                active_indices.append(sample_idx)

            if not prompt_batch:
                continue

            maybe_cuda_sync()
            outputs = llm.generate(prompt_batch, sampling_params=sampling_params)
            maybe_cuda_sync()

            for local_i, req_out in enumerate(outputs):
                sample_idx = active_indices[local_i]
                comp = req_out.outputs[0]
                output_tokens = len(comp.token_ids)
                total_output_tokens += output_tokens
                num_turns_total += 1
                histories[sample_idx].append(
                    {"role": "assistant", "content": comp.text}
                )

                if collect_responses:
                    sample_id = global_sample_offset + sample_idx
                    response_records.append(
                        {
                            "prompt_index": sample_id,
                            "sample_id": sample_id,
                            "turn_index": turn_idx,
                            "response": comp.text,
                            "num_output_tokens": output_tokens,
                        }
                    )

            if verbose:
                print(
                    f"[measure:multi] batch={batch_idx}/{len(conv_batches)} "
                    f"turn={turn_idx + 1}/{max_turns} active={len(prompt_batch)}"
                )

        global_sample_offset += len(conv_batch)

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
        "num_prompts": len(conversations),
        "batch_size": batch_size,
        "wall_time_s": wall_time_s,
        "total_output_tokens": total_output_tokens,
        "num_drafts": num_drafts_val,
        "num_draft_tokens": num_draft_tokens_val,
        "num_accepted_tokens": num_accepted_val,
        "avg_acceptance_rate": avg_acceptance_rate,
        "avg_acceptance_length": avg_acceptance_length,
        "acceptance_rate_per_pos": acceptance_rate_per_pos,
        "num_turns_total": num_turns_total,
        "avg_turns_per_sample": (
            (num_turns_total / len(conversations)) if conversations else 0.0
        ),
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

        _collect_unified_stage_metrics(result, after, before, num_drafts_val)

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
    *,
    profile_mode: str = "disabled",
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
        "spec_decode_profile_mode": profile_mode,
    }
    path = os.path.join(pair_dir, "pair_info.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[save] pair metadata: {path}")


if __name__ == "__main__":
    args = parse_args()
    if args.method == "adaptive_spechive" and not args.intermediate_model:
        raise SystemExit(
            f"--method={args.method} requires --intermediate-model"
        )
    if args.method == "pivot" and args.spechive and not args.intermediate_model:
        raise SystemExit(
            "--method=pivot with --spechive requires --intermediate-model"
        )
    random.seed(args.seed)

    # Resolve profiler mode: --profile-time is a backward-compat alias.
    profile_mode = args.spec_decode_profile_mode
    if args.profile_time and profile_mode == "disabled":
        profile_mode = "stage_cost"
    profiling_enabled = profile_mode != "disabled"

    if profiling_enabled:
        os.environ["VLLM_SPEC_VERIFY_DRAFT_MATCH"] = "1"
        print(
            f"[profiler] spec_decode_profile_mode={profile_mode}, "
            "VLLM_SPEC_VERIFY_DRAFT_MATCH=1"
        )
    else:
        os.environ["VLLM_SPEC_VERIFY_DRAFT_MATCH"] = "0"
    # Keep legacy env var in sync for ObservabilityConfig auto-promote path.
    if args.profile_time:
        os.environ["VLLM_SPEC_PROFILE_TIME"] = "1"
    else:
        os.environ["VLLM_SPEC_PROFILE_TIME"] = "0"
    if args.debug:
        os.environ["VLLM_SPEC_DIT_DEBUG"] = "1"
        os.environ["VLLM_SPEC_DIT_DEBUG_SUMMARY"] = "1"
        print("[dit_debug] VLLM_SPEC_DIT_DEBUG=1, VLLM_SPEC_DIT_DEBUG_SUMMARY=1")
    else:
        os.environ["VLLM_SPEC_DIT_DEBUG"] = "0"
        os.environ["VLLM_SPEC_DIT_DEBUG_SUMMARY"] = "0"

    batch_sizes = [int(x) for x in parse_csv_list(args.batch_sizes)]
    target_models = parse_csv_list(args.target_models)
    dataset_keys = parse_csv_list(args.datasets)
    tp_map = parse_tp_map(args.tp_map)

    print("[args] draft_model =", args.draft_model)
    print("[args] target_models =", target_models)
    print("[args] tp_map =", tp_map)
    print("[args] spec_decode_profile_mode =", profile_mode)
    print("[args] debug =", args.debug)
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
    dataset_requests = {key: get_dataset_requests(key, args) for key in dataset_keys}

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
        elif args.method == "magicdec":
            speculative_config = {
                "method": "draft_model",
                "model": args.draft_model,
                "num_speculative_tokens": args.num_spec_tokens,
                "max_model_len": args.max_model_len,
                "enforce_eager": args.enforce_eager,
                "magicdec": True,
                "magicdec_method": args.magicdec_method,
                "magicdec_kv_budget": args.magicdec_kv_budget,
            }
        elif args.method == "adaptive_spechive":
            speculative_config = {
                "method": "adaptive_spechive",
                "model": args.draft_model,
                "intermediate_model": args.intermediate_model,
                "num_speculative_tokens": args.num_spec_tokens,
                "adaptive_spechive_mode": args.adaptive_spechive_mode,
                "adaptive_spechive_num_rounds": args.round,
                "max_model_len": args.max_model_len,
                "enforce_eager": args.enforce_eager,
            }
        elif args.method == "pivot":
            # `intermediate_model` may be set for bookkeeping; only
            # `pivot_spechive` (--spechive) enables the I-stage pipeline.
            # For target-only pivot, vLLM ignores intermediate_model when building
            # the intermediate verifier (draft top-k expansion still runs).
            speculative_config = {
                "method": "pivot",
                "model": args.draft_model,
                "intermediate_model": args.intermediate_model,
                "num_speculative_tokens": args.num_spec_tokens,
                "pivot_topk_selection": args.topk_selection,
                "pivot_expansion_pct": args.expansion_pct,
                "pivot_spechive": args.spechive,
                "pivot_spechive_num_rounds": args.round,
                "max_model_len": args.max_model_len,
                "enforce_eager": args.enforce_eager,
            }
        elif args.method == "tetris":
            speculative_config = {
                "method": "draft_model",
                "model": args.draft_model,
                "num_speculative_tokens": args.num_spec_tokens,
                "max_model_len": args.max_model_len,
                "enforce_eager": args.enforce_eager,
                "tetris": True,
                "tetris_extra_proposals": args.tetris_extra_proposals,
                "tetris_turn_on_batch_size": args.tetris_turn_on_batch_size,
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

        try:
            llm_kwargs: dict[str, Any] = dict(
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
            if profiling_enabled:
                llm_kwargs["spec_decode_profile_mode"] = profile_mode
            llm = LLM(**llm_kwargs)
        except ValueError as e:
            msg = str(e)
            if (
                "Target and draft model should have the same vocabulary size" in msg
                or "Target and intermediate (verifier) model must share the same vocabulary size" in msg
                or "pivot requires intermediate output compatibility with target" in msg
            ):
                print(
                    "[skip:model-pair] incompatible speculative vocab/tokenizer "
                    f"(draft={args.draft_model}, target={target_model}): {msg}"
                )
                continue
            raise
        tokenizer = llm.get_tokenizer()
        target_rows: list[dict[str, Any]] = []
        pair_response_records: list[dict[str, Any]] = []

        for dataset_key in dataset_keys:
            request_cfg = dataset_requests[dataset_key]
            dataset_mode = str(request_cfg.get("mode", "single_turn"))
            max_new_tokens = get_dataset_max_new_tokens(dataset_key, args)

            for batch_size in batch_sizes:
                if dataset_mode == "multi_turn":
                    conversations = list(request_cfg["conversations"])
                    conversations_run, n_dataset = expand_conversations_for_batch(
                        conversations, batch_size
                    )
                    if len(conversations_run) > n_dataset:
                        print(
                            f"[dataset] tiled {n_dataset} -> {len(conversations_run)} "
                            f"conversations (batch_size={batch_size}, need >= 2*batch)"
                        )
                    print("-" * 80)
                    print(
                        f"[run] dataset={dataset_key} mode=multi_turn "
                        f"target={target_model} batch_size={batch_size} "
                        f"num_samples_dataset={n_dataset} "
                        f"num_samples_eval={len(conversations_run)} "
                        f"max_new_tokens={max_new_tokens}"
                    )
                    warmup_prompts = [conv[0] for conv in conversations_run if conv]
                    run_warmup(
                        llm=llm,
                        tokenizer=tokenizer,
                        prompts=warmup_prompts,
                        batch_size=batch_size,
                        warmup_iters=args.warmup_iters,
                        warmup_max_tokens=args.warmup_max_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        method=args.method,
                        num_spec_tokens=args.num_spec_tokens,
                        profile_time=profiling_enabled,
                    )
                    metrics = measure_dataset_multi_turn(
                        llm=llm,
                        tokenizer=tokenizer,
                        conversations=conversations_run,
                        batch_size=batch_size,
                        max_new_tokens=max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        profile_time=profiling_enabled,
                        verbose=args.verbose,
                        collect_responses=bool(
                            responses_jsonl_path and not args.no_responses
                        ),
                    )
                else:
                    prompts = list(request_cfg["prompts"])
                    prompts_run, n_dataset = expand_prompts_for_batch(prompts, batch_size)
                    if len(prompts_run) > n_dataset:
                        print(
                            f"[dataset] tiled {n_dataset} -> {len(prompts_run)} prompts "
                            f"(batch_size={batch_size}, need >= 2*batch for small sets)"
                        )
                    print("-" * 80)
                    print(
                        f"[run] dataset={dataset_key} mode=single_turn target={target_model} "
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
                        method=args.method,
                        num_spec_tokens=args.num_spec_tokens,
                        profile_time=profiling_enabled,
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
                        profile_time=profiling_enabled,
                        verbose=args.verbose,
                        draft_model=args.draft_model,
                        target_model=target_model,
                        dataset=dataset_key,
                        collect_responses=bool(
                            responses_jsonl_path and not args.no_responses
                        ),
                        num_prompts_source=n_dataset,
                    )

                if responses_jsonl_path and metrics.get("response_records"):
                    pair_response_records.extend(metrics["response_records"])

                row = {
                    "draft_model": args.draft_model,
                    "target_model": target_model,
                    "tensor_parallel_size": tp,
                    "dataset": dataset_key,
                    "mode": dataset_mode,
                    "batch_size": batch_size,
                    "num_speculative_tokens": args.num_spec_tokens,
                    "num_prompts_dataset": n_dataset,
                    "num_prompts": metrics["num_prompts"],
                    "num_turns_total": metrics.get("num_turns_total", metrics["num_prompts"]),
                    "avg_turns_per_sample": metrics.get("avg_turns_per_sample", 1.0),
                    "wall_time_s": metrics["wall_time_s"],
                    "total_output_tokens": metrics["total_output_tokens"],
                    "num_drafts": metrics["num_drafts"],
                    "num_draft_tokens": metrics["num_draft_tokens"],
                    "num_accepted_tokens": metrics["num_accepted_tokens"],
                    "avg_acceptance_rate": metrics["avg_acceptance_rate"],
                    "avg_acceptance_length": metrics["avg_acceptance_length"],
                    "acceptance_rate_per_pos": metrics["acceptance_rate_per_pos"],
                }
                if profiling_enabled:
                    # Legacy timing fields
                    row["draft_time_s"] = metrics.get("draft_time_s")
                    row["verification_time_s"] = metrics.get("verification_time_s")
                    row["avg_draft_time_s"] = metrics.get("avg_draft_time_s")
                    row["avg_verification_time_s"] = metrics.get("avg_verification_time_s")
                    row["reject_sample_time_s"] = metrics.get("reject_sample_time_s")
                    row["avg_reject_sample_time_s"] = metrics.get(
                        "avg_reject_sample_time_s"
                    )
                    # Unified profiler stage cost breakdown (ms)
                    for stage_field in (
                        "draft_forward_ms", "target_verify_ms",
                        "intermediate_verify_ms", "expand_collapse_ms",
                    ):
                        row[stage_field] = metrics.get(stage_field)
                        row[f"avg_{stage_field}"] = metrics.get(f"avg_{stage_field}")
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
                if profiling_enabled:
                    ad = row.get("avg_draft_time_s")
                    av = row.get("avg_verification_time_s")
                    if ad is not None and av is not None:
                        print(
                            f"         avg_draft_time_s={ad:.6f} "
                            f"avg_verification_time_s={av:.6f}"
                        )
                    elif row.get("draft_time_s") is None and row.get("verification_time_s") is None:
                        print(
                            "[warning] stage times are null. "
                            "Check --spec-decode-profile-mode and --disable-log-stats."
                        )
                    rs = row.get("reject_sample_time_s")
                    ars = row.get("avg_reject_sample_time_s")
                    if rs is not None:
                        ars_str = f"{ars:.6f}" if ars is not None else "N/A"
                        print(
                            f"         reject_sample_time_s={rs:.6f} "
                            f"avg_reject_sample_time_s={ars_str}"
                        )
                    # Print unified stage cost breakdown
                    stage_parts: list[str] = []
                    for sn in ("draft_forward_ms", "target_verify_ms",
                               "intermediate_verify_ms", "expand_collapse_ms"):
                        v = row.get(f"avg_{sn}")
                        if v is not None:
                            stage_parts.append(f"avg_{sn}={v:.4f}")
                    if stage_parts:
                        print(f"         {' '.join(stage_parts)}")

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
            profile_mode=profile_mode,
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
