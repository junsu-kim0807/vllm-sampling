# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


LONGCHENCH_V1_TASKS: list[str] = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
]


def _split_tasks(tasks: str) -> list[str]:
    tasks = tasks.strip()
    if tasks in {"all", "*"}:
        return list(LONGCHENCH_V1_TASKS)
    return [t.strip() for t in tasks.split(",") if t.strip()]


def _load_longbench_split(repo: str, task: str, split: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required to load LongBench. Install with: pip install datasets"
        ) from exc
    ds = load_dataset(repo, task, split=split)
    return list(ds)


def _ensure_out_dir(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _build_prompt(example: dict[str, Any]) -> str:
    # Xnhyacinth/LongBench provides:
    # - context: already includes task-specific instruction prefix
    # - question: already templated
    # - answer_prefix: e.g. "Answer:" or "Summary:"
    context = (example.get("context") or "").rstrip()
    question = (example.get("question") or "").rstrip()
    answer_prefix = (example.get("answer_prefix") or "").rstrip()
    if answer_prefix:
        return f"{context}\n\n{question}\n{answer_prefix} "
    return f"{context}\n\n{question}\n"


def _iter_chunks(items: list[Any], chunk_size: int) -> Iterable[list[Any]]:
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def _load_longbench_scorer() -> tuple[callable, callable | None]:
    """Load kvpress-baseline's LongBench scorer if available.

    Returns:
        (scorer, scorer_e_or_none)
    """
    # Resolve repo root: vllm-sampling/scripts/longbench/ -> vllm-sampling/ -> repo root
    repo_root = Path(__file__).resolve().parents[3]
    kvpress_eval = repo_root / "kvpress-baseline" / "evaluation"
    if not kvpress_eval.exists():
        raise RuntimeError(
            "kvpress-baseline evaluation package not found. "
            "Expected at: kvpress-baseline/evaluation"
        )
    import sys

    sys.path.insert(0, str(kvpress_eval))
    from benchmarks.longbench.calculate_metrics import (  # type: ignore
        calculate_metrics as longbench_scorer,
        calculate_metrics_e as longbench_scorer_e,
    )

    return longbench_scorer, longbench_scorer_e


def _maybe_make_dataframe(rows: list[dict[str, Any]]):
    """Create a pandas DataFrame if pandas is available; otherwise return None."""
    try:
        import pandas as pd  # type: ignore
    except ImportError:
        return None
    return pd.DataFrame(rows)


@dataclass
class RunConfig:
    model: str
    repo: str
    split: str
    tasks: list[str]
    out_dir: Path
    max_model_len: int | None
    gpu_memory_utilization: float
    enforce_eager: bool
    batch_size: int
    # StarKV flags (passed into vLLM EngineArgs via LLM(**kwargs))
    enable_starkv_super_cache: bool
    starkv_confidence_threshold: float | None
    starkv_offload: bool


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--repo", type=str, default="Xnhyacinth/LongBench")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--tasks", type=str, default="all", help="Comma-separated or 'all'")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--enforce-eager", action="store_true", default=False)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--skip-metrics", action="store_true", default=False)

    # StarKV / two-tier knobs
    p.add_argument("--enable-starkv-super-cache", action="store_true", default=False)
    p.add_argument("--starkv-confidence-threshold", type=float, default=None)
    p.add_argument("--starkv-offload", action="store_true", default=False)

    args = p.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path("longbench_outputs") / ts
    _ensure_out_dir(out_dir)

    tasks = _split_tasks(args.tasks)
    cfg = RunConfig(
        model=args.model,
        repo=args.repo,
        split=args.split,
        tasks=tasks,
        out_dir=out_dir,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        batch_size=args.batch_size,
        enable_starkv_super_cache=args.enable_starkv_super_cache,
        starkv_confidence_threshold=args.starkv_confidence_threshold,
        starkv_offload=args.starkv_offload,
    )

    # Initialize vLLM.
    from vllm import LLM, SamplingParams  # type: ignore

    llm_kwargs: dict[str, Any] = dict(
        enforce_eager=cfg.enforce_eager,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
    )
    if cfg.max_model_len is not None:
        llm_kwargs["max_model_len"] = cfg.max_model_len

    # Pass StarKV flags through EngineArgs.
    if cfg.enable_starkv_super_cache:
        llm_kwargs["enable_starkv_super_cache"] = True
    if cfg.starkv_confidence_threshold is not None:
        llm_kwargs["starkv_confidence_threshold"] = cfg.starkv_confidence_threshold
    if cfg.starkv_offload:
        llm_kwargs["starkv_offload"] = True

    # Optional: show reforward triggers at INFO-level if requested.
    # (The worker-side logging is controlled by env var.)
    if os.getenv("STARKV_LOG_REFORWARD") is None:
        os.environ["STARKV_LOG_REFORWARD"] = "0"

    llm = LLM(cfg.model, **llm_kwargs)

    per_task_scores: dict[str, Any] = {}
    predictions_paths: dict[str, str] = {}

    scorer, scorer_e = (None, None)
    if not args.skip_metrics:
        try:
            scorer, scorer_e = _load_longbench_scorer()
        except Exception as exc:
            print(
                "WARN: LongBench scorer could not be loaded; will skip metrics. "
                f"Reason: {exc}"
            )
            scorer, scorer_e = (None, None)

    for task in cfg.tasks:
        examples = _load_longbench_split(cfg.repo, task, cfg.split)
        prompts = [_build_prompt(ex) for ex in examples]
        # LongBench dataset provides max_new_tokens per row (constant per task in practice).
        max_new_tokens = int(examples[0].get("max_new_tokens") or 256)
        sampling = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_new_tokens,
        )

        outputs: list[str] = []
        for chunk in _iter_chunks(prompts, cfg.batch_size):
            chunk_outputs = llm.generate(chunk, sampling)
            # chunk_outputs is list[RequestOutput] aligned to inputs
            for out in chunk_outputs:
                if out.outputs:
                    outputs.append(out.outputs[0].text)
                else:
                    outputs.append("")

        pred_path = cfg.out_dir / f"predictions__{task}.csv"
        # Write predictions CSV without requiring pandas.
        fieldnames = ["task", "predicted_answer", "answers", "all_classes"]
        has_length = "length" in examples[0]
        if has_length:
            fieldnames.append("length")
        with open(pred_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for ex, pred in zip(examples, outputs):
                row = {
                    "task": task,
                    "predicted_answer": pred,
                    "answers": ex.get("answers"),
                    "all_classes": ex.get("all_classes"),
                }
                if has_length:
                    row["length"] = ex.get("length")
                w.writerow(row)
        predictions_paths[task] = str(pred_path)

        if scorer is not None:
            # The scorer expects a pandas DataFrame.
            df = _maybe_make_dataframe(
                [
                    {
                        "task": task,
                        "predicted_answer": pred,
                        "answers": ex.get("answers"),
                        "all_classes": ex.get("all_classes"),
                        **({"length": ex.get("length")} if has_length else {}),
                    }
                    for ex, pred in zip(examples, outputs)
                ]
            )
            if df is None:
                print(
                    "WARN: pandas is not installed; skipping metrics. "
                    "Install with: pip install pandas"
                )
                scorer = None
            else:
                if has_length and scorer_e is not None:
                    per_task_scores[task] = scorer_e(df)
                else:
                    per_task_scores[task] = scorer(df)

    summary = {
        "model": cfg.model,
        "repo": cfg.repo,
        "split": cfg.split,
        "tasks": cfg.tasks,
        "enable_starkv_super_cache": cfg.enable_starkv_super_cache,
        "starkv_confidence_threshold": cfg.starkv_confidence_threshold,
        "starkv_offload": cfg.starkv_offload,
        "predictions": predictions_paths,
        "scores": per_task_scores,
    }
    with open(cfg.out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved outputs to: {cfg.out_dir}")
    if per_task_scores:
        print(json.dumps(per_task_scores, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


