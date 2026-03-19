#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
JOBS_ROOT = REPO_ROOT / "scripts" / "jobs" / "spec_decode"
LOGS_ROOT = REPO_ROOT / "scripts" / "logs" / "spec_decode"
RESULTS_ROOT = REPO_ROOT / "results" / "spec_decode"

REPO_DIR = "/home/jhwoo36/scratch/vllm-sampling"
VENV_DIR = "/home/jhwoo36/scratch/venvs/vllm"

@dataclass(frozen=True)
class PairConfig:
    pair_id: str
    draft_model: str
    target_model: str
    tp_size: int
    gpu_count: int
    note: str = ""


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    max_new_tokens: int


PAIRS: list[PairConfig] = [
    PairConfig(
        pair_id="llama32_1b_to_llama32_3b",
        draft_model="meta-llama/Llama-3.2-1B-Instruct",
        target_model="meta-llama/Llama-3.2-3B-Instruct",
        tp_size=1,
        gpu_count=1,
        note="Llama small->small",
    ),
    PairConfig(
        pair_id="llama32_1b_to_llama31_8b",
        draft_model="meta-llama/Llama-3.2-1B-Instruct",
        target_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        tp_size=1,
        gpu_count=1,
        note="Llama 1B draft -> 8B target",
    ),
    PairConfig(
        pair_id="llama32_1b_to_llama31_70b",
        draft_model="meta-llama/Llama-3.2-1B-Instruct",
        target_model="meta-llama/Llama-3.1-70B-Instruct",
        tp_size=2,
        gpu_count=2,
        note="Llama 1B draft -> 70B target",
    ),
    PairConfig(
        pair_id="llama32_3b_to_llama31_8b",
        draft_model="meta-llama/Llama-3.2-3B-Instruct",
        target_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        tp_size=1,
        gpu_count=1,
        note="Llama 3B draft -> 8B target",
    ),
    PairConfig(
        pair_id="llama32_3b_to_llama31_70b",
        draft_model="meta-llama/Llama-3.2-3B-Instruct",
        target_model="meta-llama/Llama-3.1-70B-Instruct",
        tp_size=2,
        gpu_count=2,
        note="Llama 3B draft -> 70B target",
    ),
    PairConfig(
        pair_id="llama31_8b_to_llama31_70b",
        draft_model="meta-llama/Llama-3.1-8B-Instruct",
        target_model="meta-llama/Llama-3.1-70B-Instruct",
        tp_size=4,
        gpu_count=4,
        note="Llama 8B draft -> 70B target",
    ),
    PairConfig(
        pair_id="deepseekcoder_1p3b_to_6p7b",
        draft_model="deepseek-ai/deepseek-coder-1.3b-instruct",
        target_model="deepseek-ai/deepseek-coder-6.7b-instruct",
        tp_size=1,
        gpu_count=1,
        note="DeepSeek Coder 1.3B draft -> 6.7B target",
    ),
    PairConfig(
        pair_id="deepseekcoder_1p3b_to_33b",
        draft_model="deepseek-ai/deepseek-coder-1.3b-instruct",
        target_model="deepseek-ai/deepseek-coder-33b-instruct",
        tp_size=2,
        gpu_count=2,
        note="DeepSeek Coder 1.3B draft -> 33B target",
    ),
    PairConfig(
        pair_id="deepseekcoder_6p7b_to_33b",
        draft_model="deepseek-ai/deepseek-coder-6.7b-instruct",
        target_model="deepseek-ai/deepseek-coder-33b-instruct",
        tp_size=2,
        gpu_count=2,
        note="DeepSeek Coder 6.7B draft -> 33B target",
    ),
]

DATASETS: list[DatasetConfig] = [
    DatasetConfig(name="aime25", max_new_tokens=256),
    DatasetConfig(name="codeelo", max_new_tokens=1024),
]


def sanitize_for_path(s: str) -> str:
    s = s.strip().replace("/", "__")
    s = re.sub(r"[^A-Za-z0-9._+@=,]+", "_", s)
    return s.strip("._")


def pair_slug(draft_model: str, target_model: str) -> str:
    return f"{sanitize_for_path(draft_model)}__TO__{sanitize_for_path(target_model)}"


def shquote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def ensure_dirs() -> None:
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)


def infer_target_size_b(model_name: str) -> float | None:
    text = model_name.lower()

    candidates = re.findall(r"(\d+(?:\.\d+)?)b", text)
    if not candidates:
        return None

    values = [float(x) for x in candidates]
    if not values:
        return None

    return max(values)


def time_limit_for_dataset(pair: PairConfig, dataset: DatasetConfig) -> str:
    if dataset.name == "aime25":
        return "01:30:00"

    if dataset.name == "codeelo":
        size_b = infer_target_size_b(pair.target_model)
        if size_b is None:
            return "03:00:00"
        if abs(size_b - 70.0) < 1e-9:
            return "10:00:00"
        if 30.0 <= size_b <= 40.0:
            return "06:00:00"
        return "03:00:00"

    return "03:00:00"


def job_header(job_name: str, gpu_count: int, time_limit: str, log_dir: Path) -> str:
    log_dir.mkdir(parents=True, exist_ok=True)
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --account=rrg-pnair_gpu
#SBATCH --qos=rrg-pnair
#SBATCH --gres=gpu:h100:{gpu_count}
#SBATCH --mem-per-gpu=80G
#SBATCH --time={time_limit}
#SBATCH --output={log_dir / (job_name + ".out")}
#SBATCH --error={log_dir / (job_name + ".err")}

set -euo pipefail

module load python/3.12 cuda/12.9 arrow/21.0.0

REPO_DIR={shquote(REPO_DIR)}
VENV_DIR={shquote(VENV_DIR)}

cd "${{REPO_DIR}}"
source "${{VENV_DIR}}/bin/activate"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export HF_TOKEN="$HF_TOKEN"
export VLLM_WORKER_MULTIPROC_METHOD="${{VLLM_WORKER_MULTIPROC_METHOD:-spawn}}"
export HF_HOME=/home/jhwoo36/scratch/.cache
export TRANSFORMERS_CACHE=/home/jhwoo36/scratch/.cache/transformers
export HUGGINGFACE_HUB_CACHE=/home/jhwoo36/scratch/.cache/huggingface_hub
export HF_DATASETS_CACHE=/home/jhwoo36/scratch/.cache/datasets
"""


def build_python_command(
    pair: PairConfig,
    dataset: DatasetConfig,
    batch_sizes: str,
    gpu_mem_util: float,
    max_model_len: int,
    dtype: str,
    seed: int,
    warmup_iters: int,
    warmup_max_tokens: int,
    verbose: bool,
) -> str:
    root_for_dataset = RESULTS_ROOT / dataset.name
    slug = pair_slug(pair.draft_model, pair.target_model)

    args: list[str] = [
        "python -u scripts/run_spec_decode_metrics.py",
        f"--draft-model {shquote(pair.draft_model)}",
        f"--target-models {shquote(pair.target_model)}",
        f"--tp-map {shquote(f'{pair.target_model}={pair.tp_size}')}",
        f"--datasets {dataset.name}",
        f"--batch-sizes {batch_sizes}",
        "--num-spec-tokens 7",
        f"--gpu-memory-utilization {gpu_mem_util:.2f}",
        f"--max-model-len {max_model_len}",
        f"--dtype {dtype}",
        f"--seed {seed}",
        f"--warmup-iters {warmup_iters}",
        f"--warmup-max-tokens {warmup_max_tokens}",
        f"--aime-max-new-tokens {dataset.max_new_tokens if dataset.name == 'aime25' else 256}",
        f"--codeelo-max-new-tokens {dataset.max_new_tokens if dataset.name == 'codeelo' else 1024}",
        f"--results-root {shquote(str(root_for_dataset))}",
        f"--results-csv {shquote(f'aggregate__{slug}.csv')}",
        f"--results-jsonl {shquote(f'aggregate__{slug}.jsonl')}",
        "--profile-time",
        "--trust-remote-code",
        "--enable-chunked-prefill",
    ]

    if verbose:
        args.append("--verbose")

    return " \\\n  ".join(args)


def render_job_script(
    pair: PairConfig,
    dataset: DatasetConfig,
    batch_sizes: str,
    gpu_mem_util: float,
    max_model_len: int,
    dtype: str,
    seed: int,
    warmup_iters: int,
    warmup_max_tokens: int,
    verbose: bool,
) -> str:
    slug = pair_slug(pair.draft_model, pair.target_model)
    job_name = f"spec_{pair.pair_id}_{dataset.name}"

    job_dir = JOBS_ROOT / pair.pair_id
    log_dir = LOGS_ROOT / pair.pair_id
    result_dataset_root = RESULTS_ROOT / dataset.name
    pair_result_dir = result_dataset_root / slug

    aggregate_csv = result_dataset_root / f"aggregate__{slug}.csv"
    aggregate_jsonl = result_dataset_root / f"aggregate__{slug}.jsonl"

    time_limit = time_limit_for_dataset(pair, dataset)

    header = job_header(
        job_name=job_name,
        gpu_count=pair.gpu_count,
        time_limit=time_limit,
        log_dir=log_dir,
    )

    command = build_python_command(
        pair=pair,
        dataset=dataset,
        batch_sizes=batch_sizes,
        gpu_mem_util=gpu_mem_util,
        max_model_len=max_model_len,
        dtype=dtype,
        seed=seed,
        warmup_iters=warmup_iters,
        warmup_max_tokens=warmup_max_tokens,
        verbose=verbose,
    )

    body = f"""
DATASET_NAME={shquote(dataset.name)}
PAIR_ID={shquote(pair.pair_id)}
PAIR_SLUG={shquote(slug)}

RESULT_DATASET_ROOT={shquote(str(result_dataset_root))}
PAIR_RESULT_DIR={shquote(str(pair_result_dir))}
AGGREGATE_CSV={shquote(str(aggregate_csv))}
AGGREGATE_JSONL={shquote(str(aggregate_jsonl))}

mkdir -p "${{RESULT_DATASET_ROOT}}" "${{PAIR_RESULT_DIR}}"

echo "============================================================"
echo "DATASET: ${{DATASET_NAME}}"
echo "PAIR_ID: ${{PAIR_ID}}"
echo "PAIR_SLUG: ${{PAIR_SLUG}}"
echo "DRAFT_MODEL: {pair.draft_model}"
echo "TARGET_MODEL: {pair.target_model}"
echo "TP_SIZE: {pair.tp_size}"
echo "TIME_LIMIT: {time_limit}"
echo "RESULT_DATASET_ROOT: ${{RESULT_DATASET_ROOT}}"
echo "PAIR_RESULT_DIR: ${{PAIR_RESULT_DIR}}"
echo "PWD: $(pwd)"
echo "PYTHON: $(which python)"
echo "============================================================"

{command}

if [[ -f "${{AGGREGATE_CSV}}" ]]; then
  mv -f "${{AGGREGATE_CSV}}" "${{PAIR_RESULT_DIR}}/aggregate.csv"
fi

if [[ -f "${{AGGREGATE_JSONL}}" ]]; then
  mv -f "${{AGGREGATE_JSONL}}" "${{PAIR_RESULT_DIR}}/aggregate.jsonl"
fi

echo "Saved final outputs under: ${{PAIR_RESULT_DIR}}"
"""
    return header + "\n" + body.strip() + "\n"


def write_job_script(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def filter_by_attr(items: list, wanted: set[str], attr: str) -> list:
    if not wanted:
        return items
    out = []
    for item in items:
        if getattr(item, attr) in wanted:
            out.append(item)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=[])
    parser.add_argument("--pairs", nargs="*", default=[])
    parser.add_argument("--batch-sizes", default="1")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--warmup-max-tokens", type=int, default=32)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    datasets = filter_by_attr(DATASETS, set(args.datasets), "name")
    pairs = filter_by_attr(PAIRS, set(args.pairs), "pair_id")

    num_written = 0

    for pair in pairs:
        for dataset in datasets:
            script_path = JOBS_ROOT / pair.pair_id / f"{dataset.name}.slurm"
            script_text = render_job_script(
                pair=pair,
                dataset=dataset,
                batch_sizes=args.batch_sizes,
                gpu_mem_util=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                dtype=args.dtype,
                seed=args.seed,
                warmup_iters=args.warmup_iters,
                warmup_max_tokens=args.warmup_max_tokens,
                verbose=args.verbose,
            )
            write_job_script(script_path, script_text)
            num_written += 1
            print(f"Wrote {script_path}")

    print(f"Generated {num_written} job scripts.")


if __name__ == "__main__":
    main()