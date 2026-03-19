#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate Slurm jobs under scripts/jobs/spec_decode/ for run_spec_decode_metrics.

  python scripts/generate_sd_script.py [--pairs ...] [--datasets ...]
  python scripts/generate_sd_script.py --test [--test-pairs ...] [--test-samples 5]
  python scripts/generate_sd_script.py --batch --datasets gov_report qmsum

With --batch, it generates AR/speculative decoding/eagle3 jobs for batch sizes
1/4/16/64/256 with fixed Slurm time limits and writes a submit shell.

LongBench-v1: gov_report, qmsum (THUDM/LongBench). Set HF_TOKEN before submit.
"""

from __future__ import annotations

import argparse
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
JOBS_ROOT = REPO_ROOT / "scripts" / "jobs" / "spec_decode"
LOGS_ROOT = REPO_ROOT / "scripts" / "logs" / "spec_decode"
RESULTS_ROOT = REPO_ROOT / "results" / "spec_decode"

# Override with env or edit for your cluster:
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
    # PairConfig(
    #     pair_id="llama32_1b_to_llama32_3b",
    #     draft_model="meta-llama/Llama-3.2-1B-Instruct",
    #     target_model="meta-llama/Llama-3.2-3B-Instruct",
    #     tp_size=1,
    #     gpu_count=1,
    #     note="Llama small->small",
    # ),
    # PairConfig(
    #     pair_id="llama32_1b_to_llama31_8b",
    #     draft_model="meta-llama/Llama-3.2-1B-Instruct",
    #     target_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    #     tp_size=1,
    #     gpu_count=1,
    #     note="Llama 1B draft -> 8B target",
    # ),
    PairConfig(
        pair_id="llama32_1b_to_llama31_70b",
        draft_model="meta-llama/Llama-3.2-1B-Instruct",
        target_model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        tp_size=2,
        gpu_count=2,
        note="Llama 1B draft -> 70B target",
    ),
    # PairConfig(
    #     pair_id="llama32_3b_to_llama31_8b",
    #     draft_model="meta-llama/Llama-3.2-3B-Instruct",
    #     target_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    #     tp_size=1,
    #     gpu_count=1,
    #     note="Llama 3B draft -> 8B target",
    # ),
    PairConfig(
        pair_id="llama32_3b_to_llama31_70b",
        draft_model="meta-llama/Llama-3.2-3B-Instruct",
        target_model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        tp_size=2,
        gpu_count=2,
        note="Llama 3B draft -> 70B target",
    ),
    # PairConfig(
    #     pair_id="llama31_8b_to_llama31_70b",
    #     draft_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    #     target_model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    #     tp_size=4,
    #     gpu_count=4,
    #     note="Llama 8B draft -> 70B target",
    # ),
    # PairConfig(
    #     pair_id="deepseekcoder_1p3b_to_6p7b",
    #     draft_model="deepseek-ai/deepseek-coder-1.3b-instruct",
    #     target_model="deepseek-ai/deepseek-coder-6.7b-instruct",
    #     tp_size=1,
    #     gpu_count=1,
    #     note="DeepSeek Coder 1.3B draft -> 6.7B target",
    # ),
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
    # Qwen3 speculative decoding pairs
    # PairConfig(
    #     pair_id="qwen3_0p6b_to_qwen3_4b",
    #     draft_model="Qwen/Qwen3-0.6B",
    #     target_model="Qwen/Qwen3-4B",
    #     tp_size=1,
    #     gpu_count=1,
    #     note="Qwen3 0.6B draft -> 4B target",
    # ),
    # PairConfig(
    #     pair_id="qwen3_0p6b_to_qwen3_8b",
    #     draft_model="Qwen/Qwen3-0.6B",
    #     target_model="Qwen/Qwen3-8B",
    #     tp_size=1,
    #     gpu_count=1,
    #     note="Qwen3 0.6B draft -> 8B target",
    # ),
    PairConfig(
        pair_id="qwen3_0p6b_to_qwen3_30b_a3b",
        draft_model="Qwen/Qwen3-0.6B",
        target_model="Qwen/Qwen3-30B-A3B",
        tp_size=2,
        gpu_count=2,
        note="Qwen3 0.6B draft -> 30B-A3B target",
    ),
    # PairConfig(
    #     pair_id="qwen3_4b_to_qwen3_8b",
    #     draft_model="Qwen/Qwen3-4B",
    #     target_model="Qwen/Qwen3-8B",
    #     tp_size=1,
    #     gpu_count=1,
    #     note="Qwen3 4B draft -> 8B target",
    # ),
    PairConfig(
        pair_id="qwen3_4b_to_qwen3_30b_a3b",
        draft_model="Qwen/Qwen3-4B",
        target_model="Qwen/Qwen3-30B-A3B",
        tp_size=2,
        gpu_count=2,
        note="Qwen3 4B draft -> 30B-A3B target",
    ),
    # PairConfig(
    #     pair_id="qwen3_8b_to_qwen3_30b_a3b",
    #     draft_model="Qwen/Qwen3-8B",
    #     target_model="Qwen/Qwen3-30B-A3B",
    #     tp_size=2,
    #     gpu_count=2,
    #     note="Qwen3 8B draft -> 30B-A3B target",
    # ),
]

DATASETS: list[DatasetConfig] = [
    DatasetConfig(name="aime25", max_new_tokens=256),
    DatasetConfig(name="codeelo", max_new_tokens=1024),
    DatasetConfig(name="gov_report", max_new_tokens=512),
    DatasetConfig(name="qmsum", max_new_tokens=512),
]

DEFAULT_TEST_PAIR_IDS = frozenset({"llama32_1b_to_llama31_8b"})


def sanitize_for_path(s: str) -> str:
    s = s.strip().replace("/", "__")
    s = re.sub(r"[^A-Za-z0-9._+@=,]+", "_", s)
    return s.strip("._")


def pair_slug(draft_model: str, target_model: str) -> str:
    return f"{sanitize_for_path(draft_model)}__TO__{sanitize_for_path(target_model)}"


def batch_tag(batch_sizes: str) -> str:
    """e.g. '1,4,8' -> 'b1_4_8' for use in result directory names."""
    parts = [p.strip() for p in batch_sizes.split(",") if p.strip()]
    return "b" + "_".join(parts)


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
    return max(values) if values else None


def time_limit_for_dataset(
    pair: PairConfig, dataset: DatasetConfig, *, test: bool
) -> str:
    if test:
        return "01:30:00"

    if dataset.name == "aime25":
        return "01:30:00"

    if dataset.name in ("gov_report", "qmsum"):
        size_b = infer_target_size_b(pair.target_model)
        if size_b is not None and size_b >= 60.0:
            return "12:00:00"
        if size_b is not None and size_b >= 30.0:
            return "08:00:00"
        return "04:00:00"

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
# export HF_TOKEN before sbatch if models are gated
export HF_TOKEN="${{HF_TOKEN:-}}"
export VLLM_WORKER_MULTIPROC_METHOD="${{VLLM_WORKER_MULTIPROC_METHOD:-spawn}}"
export HF_HOME="${{HF_HOME:-/home/jhwoo36/scratch/.cache}}"
export TRANSFORMERS_CACHE="${{TRANSFORMERS_CACHE:-/home/jhwoo36/scratch/.cache/transformers}}"
export HUGGINGFACE_HUB_CACHE="${{HUGGINGFACE_HUB_CACHE:-/home/jhwoo36/scratch/.cache/huggingface_hub}}"
export HF_DATASETS_CACHE="${{HF_DATASETS_CACHE:-/home/jhwoo36/scratch/.cache/datasets}}"
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
    num_spec_tokens: int,
    verbose: bool,
    *,
    test: bool,
    test_samples: int,
    method: str = "speculative",
    eagle_model: str | None = None,
    eagle_draft_tp: int | None = None,
) -> str:
    tag = batch_tag(batch_sizes)
    # results/spec_decode/<method>/<tag>/<dataset>/
    root_for_dataset = RESULTS_ROOT / method / tag / dataset.name
    slug = pair_slug(pair.draft_model, pair.target_model)

    parts: list[str] = [
        "python -u scripts/run_spec_decode_metrics.py",
        f"--draft-model {shquote(pair.draft_model)}",
        f"--target-models {shquote(pair.target_model)}",
        f"--tp-map {shquote(f'{pair.target_model}={pair.tp_size}')}",
        f"--datasets {dataset.name}",
        f"--batch-sizes {batch_sizes}",
        f"--num-spec-tokens {num_spec_tokens}",
        f"--gpu-memory-utilization {gpu_mem_util:.2f}",
        f"--max-model-len {max_model_len}",
        f"--dtype {dtype}",
        f"--seed {seed}",
        f"--warmup-iters {warmup_iters}",
        f"--warmup-max-tokens {warmup_max_tokens}",
        f"--results-root {shquote(str(root_for_dataset))}",
        f"--results-csv {shquote(f'aggregate__{slug}.csv')}",
        f"--results-jsonl {shquote(f'aggregate__{slug}.jsonl')}",
        "--profile-time",
        "--trust-remote-code",
        "--enable-chunked-prefill",
        f"--method {method}",
    ]

    if method == "eagle3":
        if not eagle_model:
            raise SystemExit("--method=eagle3 requires --eagle-model from generator")
        parts.append(f"--eagle-model {shquote(eagle_model)}")
        parts.append(f"--eagle-draft-tp {eagle_draft_tp if eagle_draft_tp is not None else pair.tp_size}")

    if dataset.name == "aime25":
        parts.append(f"--aime-max-new-tokens {dataset.max_new_tokens}")
        parts.append("--codeelo-max-new-tokens 1024")
        parts.append("--gov-report-max-new-tokens 512")
        parts.append("--qmsum-max-new-tokens 512")
    elif dataset.name == "codeelo":
        parts.append("--aime-max-new-tokens 256")
        parts.append(f"--codeelo-max-new-tokens {dataset.max_new_tokens}")
        parts.append("--gov-report-max-new-tokens 512")
        parts.append("--qmsum-max-new-tokens 512")
    elif dataset.name == "gov_report":
        parts.append("--aime-max-new-tokens 256")
        parts.append("--codeelo-max-new-tokens 1024")
        parts.append(f"--gov-report-max-new-tokens {dataset.max_new_tokens}")
        parts.append("--qmsum-max-new-tokens 512")
    else:
        parts.append("--aime-max-new-tokens 256")
        parts.append("--codeelo-max-new-tokens 1024")
        parts.append("--gov-report-max-new-tokens 512")
        parts.append(f"--qmsum-max-new-tokens {dataset.max_new_tokens}")

    if test:
        ts = test_samples
        parts.append(f"--max-samples-aime25 {ts}")
        parts.append(f"--max-samples-codeelo {ts}")
        parts.append(f"--max-samples-gov-report {ts}")
        parts.append(f"--max-samples-qmsum {ts}")

    if verbose:
        parts.append("--verbose")

    return " \\\n  ".join(parts)


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
    num_spec_tokens: int,
    verbose: bool,
    *,
    test: bool,
    test_samples: int,
    method: str = "speculative",
    eagle_model: str | None = None,
    eagle_draft_tp: int | None = None,
    time_limit_override: str | None = None,
) -> str:
    slug = pair_slug(pair.draft_model, pair.target_model)
    tag = batch_tag(batch_sizes)
    suffix = "_test" if test else ""
    job_name = f"spec_{pair.pair_id}_{dataset.name}{suffix}_{tag}"

    pair_subdir = Path("test") / pair.pair_id if test else Path(pair.pair_id)
    job_dir = JOBS_ROOT / pair_subdir
    log_dir = LOGS_ROOT / pair_subdir
    # Keep job script and run_spec_decode_metrics aligned:
    # results/spec_decode/<method>/<tag>/<dataset>/
    result_dataset_root = RESULTS_ROOT / method / tag / dataset.name
    pair_result_dir = result_dataset_root / slug

    aggregate_csv = result_dataset_root / f"aggregate__{slug}.csv"
    aggregate_jsonl = result_dataset_root / f"aggregate__{slug}.jsonl"

    time_limit = (
        time_limit_override
        if time_limit_override is not None
        else time_limit_for_dataset(pair, dataset, test=test)
    )

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
        num_spec_tokens=num_spec_tokens,
        verbose=verbose,
        test=test,
        test_samples=test_samples,
        method=method,
        eagle_model=eagle_model,
        eagle_draft_tp=eagle_draft_tp,
    )

    body = f"""
DATASET_NAME={shquote(dataset.name)}
PAIR_ID={shquote(pair.pair_id)}
PAIR_SLUG={shquote(slug)}
TEST_MODE={shquote(str(test))}

RESULT_DATASET_ROOT={shquote(str(result_dataset_root))}
PAIR_RESULT_DIR={shquote(str(pair_result_dir))}
AGGREGATE_CSV={shquote(str(aggregate_csv))}
AGGREGATE_JSONL={shquote(str(aggregate_jsonl))}

mkdir -p "${{RESULT_DATASET_ROOT}}" "${{PAIR_RESULT_DIR}}"

echo "============================================================"
echo "DATASET: ${{DATASET_NAME}}"
echo "PAIR_ID: ${{PAIR_ID}}"
echo "TEST_MODE: ${{TEST_MODE}}"
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
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def filter_by_attr(items: list, wanted: set[str], attr: str) -> list:
    if not wanted:
        return list(items)
    out = []
    for item in items:
        if getattr(item, attr) in wanted:
            out.append(item)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=[], help="Subset of dataset names")
    parser.add_argument("--pairs", nargs="*", default=[], help="Subset of pair_id values")
    parser.add_argument("--batch-sizes", default="1")
    parser.add_argument(
        "--batch",
        action="store_true",
        help=(
            "Generate AR/speculative decoding/eagle3 jobs for batch sizes "
            "(1,4,16,64,256) with fixed time limits."
        ),
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--warmup-max-tokens", type=int, default=32)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--test",
        action="store_true",
        help=(
            "Write smoke-test jobs under jobs/spec_decode/test/<pair_id>/. "
            "Default pairs: llama32_1b_to_llama31_8b. Caps samples per dataset, "
            "shorter time, num_spec_tokens=3, warmup 1."
        ),
    )
    parser.add_argument(
        "--test-pairs",
        nargs="*",
        default=[],
        help="With --test: pair_id list (default: llama32_1b_to_llama31_8b).",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=5,
        help="With --test: --max-samples-* per dataset (default 5).",
    )
    parser.add_argument(
        "--test-batch-sizes",
        default="1",
        help="With --test: batch sizes (default 1).",
    )
    parser.add_argument(
        "--test-max-model-len",
        type=int,
        default=None,
        help="With --test: override --max-model-len (default: same as --max-model-len).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    datasets = filter_by_attr(DATASETS, set(args.datasets), "name")

    if args.batch:
        if args.test:
            raise SystemExit("Use either --batch or --test (not both).")
        if not datasets:
            raise SystemExit(
                "With --batch: please pass at least one --datasets value. "
                f"Known: {[d.name for d in DATASETS]}"
            )

        batch_sizes_list = [1, 4, 16, 64, 256, 512, 1024]

        def time_limit_for_batch(bs: int) -> str:
            if bs == 1:
                return "05:00:00"
            if bs == 4:
                return "04:00:00"
            if bs == 16:
                return "03:00:00"
            if bs == 64:
                return "02:00:00"
            if bs in (256, 512, 1024):
                return "00:30:00"
            raise ValueError(f"Unsupported batch size: {bs}")

        # --- model selection ---
        llama33_70b = "meta-llama/Llama-3.3-70B-Instruct"
        llama31_70b_old = "meta-llama/Meta-Llama-3.1-70B-Instruct"
        qwen30b_a3b = "Qwen/Qwen3-30B-A3B-Instruct-2507"

        # You may need to adjust these two env overrides to match your HF setup.
        eagle_llama33_speculator = os.environ.get(
            "EAGLE3_LLAMA33_70B_SPECULATOR",
            "RedHatAI/Llama-3.3-70B-Instruct-speculator.eagle3",
        )
        eagle_qwen30b_a3b_speculator = os.environ.get(
            "EAGLE3_QWEN30B_A3B_SPECULATOR",
            "RedHatAI/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3",
        )

        # AR pseudo-pairs
        ar_pairs: list[PairConfig] = [
            PairConfig(
                pair_id="ar_llama33_70b",
                draft_model=llama33_70b,
                target_model=llama33_70b,
                tp_size=4,
                gpu_count=4,
                note="AR only (no speculative decoding)",
            ),
            PairConfig(
                pair_id="ar_qwen30b_a3b",
                draft_model=qwen30b_a3b,
                target_model=qwen30b_a3b,
                tp_size=2,
                gpu_count=2,
                note="AR only (no speculative decoding)",
            ),
            PairConfig(
                pair_id="ar_deepseekcoder_33b",
                draft_model="deepseek-ai/deepseek-coder-33b-instruct",
                target_model="deepseek-ai/deepseek-coder-33b-instruct",
                tp_size=2,
                gpu_count=2,
                note="AR only (no speculative decoding)",
            ),
        ]

        # EAGLE3 pseudo-pairs
        eagle_pairs: list[PairConfig] = [
            PairConfig(
                pair_id="eagle3_llama33_70b",
                draft_model=llama33_70b,
                target_model=llama33_70b,
                tp_size=4,
                gpu_count=4,
                note="EAGLE3 (eagle3 method)",
            ),
            PairConfig(
                pair_id="eagle3_qwen30b_a3b",
                draft_model=qwen30b_a3b,
                target_model=qwen30b_a3b,
                tp_size=2,
                gpu_count=2,
                note="EAGLE3 (eagle3 method)",
            ),
        ]

        # Speculative decoding pairs, but swap 70B target to Llama 3.3.
        speculative_pairs: list[PairConfig] = []
        for pair in PAIRS:
            if pair.target_model == llama31_70b_old:
                new_pair_id = pair.pair_id.replace("llama31_70b", "llama33_70b")
                speculative_pairs.append(
                    PairConfig(
                        pair_id=new_pair_id,
                        draft_model=pair.draft_model,
                        target_model=llama33_70b,
                        tp_size=4,
                        gpu_count=4,
                        note=pair.note,
                    )
                )
            else:
                speculative_pairs.append(pair)

        # Additional Qwen speculative experiments requested by user.
        speculative_pairs.extend(
            [
                PairConfig(
                    pair_id="qwen25_0p5b_to_qwen3_4b_instruct_2507",
                    draft_model="Qwen/Qwen2.5-0.5B-Instruct",
                    target_model="Qwen/Qwen3-4B-Instruct-2507",
                    tp_size=1,
                    gpu_count=1,
                    note="Qwen2.5 0.5B draft -> Qwen3 4B Instruct 2507",
                ),
                PairConfig(
                    pair_id="qwen25_0p5b_to_qwen3_30b_a3b_instruct_2507",
                    draft_model="Qwen/Qwen2.5-0.5B-Instruct",
                    target_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
                    tp_size=2,
                    gpu_count=2,
                    note="Qwen2.5 0.5B draft -> Qwen3 30B-A3B Instruct 2507",
                ),
                PairConfig(
                    pair_id="qwen3_0p6b_to_qwen3_4b_instruct_2507",
                    draft_model="Qwen/Qwen3-0.6B",
                    target_model="Qwen/Qwen3-4B-Instruct-2507",
                    tp_size=1,
                    gpu_count=1,
                    note="Qwen3 0.6B draft -> Qwen3 4B Instruct 2507",
                ),
                PairConfig(
                    pair_id="qwen3_0p6b_to_qwen3_30b_a3b_instruct_2507",
                    draft_model="Qwen/Qwen3-0.6B",
                    target_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
                    tp_size=2,
                    gpu_count=2,
                    note="Qwen3 0.6B draft -> Qwen3 30B-A3B Instruct 2507",
                ),
            ]
        )

        # --- common tuning (prod defaults from existing generator) ---
        test = False
        test_samples = 5
        warmup_iters = args.warmup_iters
        warmup_max_tokens = args.warmup_max_tokens
        num_spec = 7
        max_model_len = args.max_model_len

        written_scripts: list[Path] = []
        num_written = 0

        def _write_one(
            *,
            pair: PairConfig,
            dataset: DatasetConfig,
            bs: int,
            method: str,
            eagle_model: str | None,
            eagle_draft_tp: int | None = None,
            time_limit_override: str,
        ) -> None:
            batch_sizes_str = str(bs)
            script_path = JOBS_ROOT / pair.pair_id / f"{dataset.name}_b{bs}.slurm"
            script_text = render_job_script(
                pair=pair,
                dataset=dataset,
                batch_sizes=batch_sizes_str,
                gpu_mem_util=args.gpu_memory_utilization,
                max_model_len=max_model_len,
                dtype=args.dtype,
                seed=args.seed,
                warmup_iters=warmup_iters,
                warmup_max_tokens=warmup_max_tokens,
                num_spec_tokens=num_spec,
                verbose=args.verbose,
                test=test,
                test_samples=test_samples,
                method=method,
                eagle_model=eagle_model,
                eagle_draft_tp=eagle_draft_tp,
                time_limit_override=time_limit_override,
            )
            write_job_script(script_path, script_text)
            written_scripts.append(script_path)

        for bs in batch_sizes_list:
            tl = time_limit_for_batch(bs)
            for dataset in datasets:
                # AR
                for pair in ar_pairs:
                    _write_one(
                        pair=pair,
                        dataset=dataset,
                        bs=bs,
                        method="ar",
                        eagle_model=None,
                        time_limit_override=tl,
                    )

                # Speculative decoding (draft_model)
                for pair in speculative_pairs:
                    _write_one(
                        pair=pair,
                        dataset=dataset,
                        bs=bs,
                        method="speculative",
                        eagle_model=None,
                        time_limit_override=tl,
                    )

                # Eagle3
                for pair in eagle_pairs:
                    eagle_model = (
                        eagle_llama33_speculator
                        if pair.target_model == llama33_70b
                        else eagle_qwen30b_a3b_speculator
                    )
                    _write_one(
                        pair=pair,
                        dataset=dataset,
                        bs=bs,
                        method="eagle3",
                        eagle_model=eagle_model,
                        eagle_draft_tp=1,
                        time_limit_override=tl,
                    )

        num_written = len(written_scripts)
        print(f"[batch] Generated {num_written} job scripts.")

        submit_path = REPO_ROOT / "scripts" / "submit_spec_decode_batch_scaling.sh"
        with open(submit_path, "w", encoding="utf-8") as f:
            f.write("#!/usr/bin/env bash\n")
            f.write("set -euo pipefail\n\n")
            f.write(f'REPO_DIR="{REPO_ROOT}"\n\n')
            for sp in written_scripts:
                f.write(f'echo "sbatch {sp}"\n')
                f.write(f"sbatch {sp}\n")
        submit_path.chmod(submit_path.stat().st_mode | stat.S_IXUSR)
        print(f"[batch] Submit script: {submit_path}")
        return

    if args.test:
        pair_ids = set(args.test_pairs) if args.test_pairs else set(DEFAULT_TEST_PAIR_IDS)
        pairs = filter_by_attr(PAIRS, pair_ids, "pair_id")
        if not pairs:
            raise SystemExit(
                f"--test: no pairs matched {pair_ids!r}. Known: "
                + ", ".join(p.pair_id for p in PAIRS)
            )
        test = True
        test_samples = args.test_samples
        batch_sizes = args.test_batch_sizes
        warmup_iters = 1
        warmup_max_tokens = 16
        num_spec = 3
        max_model_len = args.test_max_model_len or args.max_model_len
    else:
        pairs = filter_by_attr(PAIRS, set(args.pairs), "pair_id")
        test = False
        test_samples = 5
        batch_sizes = args.batch_sizes
        warmup_iters = args.warmup_iters
        warmup_max_tokens = args.warmup_max_tokens
        num_spec = 7
        max_model_len = args.max_model_len

    num_written = 0
    for pair in pairs:
        for dataset in datasets:
            pair_subdir = Path("test") / pair.pair_id if test else Path(pair.pair_id)
            script_path = JOBS_ROOT / pair_subdir / f"{dataset.name}.slurm"
            script_text = render_job_script(
                pair=pair,
                dataset=dataset,
                batch_sizes=batch_sizes,
                gpu_mem_util=args.gpu_memory_utilization,
                max_model_len=max_model_len,
                dtype=args.dtype,
                seed=args.seed,
                warmup_iters=warmup_iters,
                warmup_max_tokens=warmup_max_tokens,
                num_spec_tokens=num_spec,
                verbose=args.verbose,
                test=test,
                test_samples=test_samples,
            )
            write_job_script(script_path, script_text)
            num_written += 1
            print(f"Wrote {script_path}")

    mode = "test" if test else "prod"
    print(f"Generated {num_written} job scripts ({mode}).")


if __name__ == "__main__":
    main()
