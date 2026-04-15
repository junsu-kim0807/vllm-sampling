#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate Slurm jobs under scripts/jobs/spec_decode/ for run_spec_decode_metrics.

Examples:
  python scripts/generate_sd_script.py [--pairs ...] [--datasets ...]
  python scripts/generate_sd_script.py --test [--test-pairs ...] [--test-samples 5]
  python scripts/generate_sd_script.py --batch --datasets gov_report qmsum
  python scripts/generate_sd_script.py --draft --datasets codeelo
  python scripts/generate_sd_script.py --verify --datasets aime25

Modes:
  --eager  : add --enforce-eager to generated metrics command (combine with any mode)
  default  : generate regular jobs from PAIRS
  --test   : smoke test jobs
  --batch  : vary batch size
  --length : fixed batch 256, sweep num_spec_tokens (5,7,9,11) like --batch
  --ablation: pivot only; sweep topk x expansion_pct and num_spec_tokens 3,7,11
  --draft  : vary num_spec_tokens while varying draft model, fixed target
  --verify : vary num_spec_tokens while varying target model along an ordered model chain

LongBench-v1: gov_report, qmsum (THUDM/LongBench). Set HF_TOKEN before submit.
"""

from __future__ import annotations

import argparse
import getpass
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
USER_NAME = os.environ.get("USER") or getpass.getuser()

if USER_NAME == "junsuk87":
    REPO_DIR = "/scratch/junsuk87/vllm-sampling"
    VENV_DIR = "/scratch/junsuk87/venvs/vllm"
    # REPO_DIR = "/project/def-pnair/junsu/kv_cache/vllm-sampling"
    # VENV_DIR = "/project/def-pnair/junsu/kv_cache"

elif USER_NAME == "jhwoo36":
    REPO_DIR = "/home/jhwoo36/scratch/vllm-sampling"
    VENV_DIR = "/home/jhwoo36/scratch/venvs/vllm"
else:
    REPO_DIR = str(REPO_ROOT)
    VENV_DIR = str(REPO_ROOT / ".venv")

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


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_name: str
    tp_size: int
    gpu_count: int
    note: str = ""


@dataclass(frozen=True)
class BatchSpecPairConfig:
    pair_id: str
    draft_key: str
    target_key: str
    tp_size: int
    gpu_count: int
    note: str = ""
    pivot_intermediate_key: str | None = None


MODEL_SPECS: dict[str, ModelSpec] = {
    "llama32_1b": ModelSpec(
        key="llama32_1b",
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        tp_size=1,
        gpu_count=1,
        note="Llama 3.2 1B",
    ),
    "llama32_3b": ModelSpec(
        key="llama32_3b",
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        tp_size=1,
        gpu_count=1,
        note="Llama 3.2 3B",
    ),
    "llama31_8b": ModelSpec(
        key="llama31_8b",
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        tp_size=1,
        gpu_count=1,
        note="Llama 3.1 8B",
    ),
    "llama31_70b": ModelSpec(
        key="llama31_70b",
        model_name="meta-llama/Meta-Llama-3.1-70B-Instruct",
        tp_size=4,
        gpu_count=4,
        note="Llama 3.1 70B",
    ),
    "llama33_70b": ModelSpec(
        key="llama33_70b",
        model_name="meta-llama/Llama-3.3-70B-Instruct",
        tp_size=4,
        gpu_count=4,
        note="Llama 3.3 70B",
    ),
    "deepseekcoder_1p3b": ModelSpec(
        key="deepseekcoder_1p3b",
        model_name="deepseek-ai/deepseek-coder-1.3b-instruct",
        tp_size=1,
        gpu_count=1,
        note="DeepSeek Coder 1.3B",
    ),
    "deepseekcoder_6p7b": ModelSpec(
        key="deepseekcoder_6p7b",
        model_name="deepseek-ai/deepseek-coder-6.7b-instruct",
        tp_size=1,
        gpu_count=1,
        note="DeepSeek Coder 6.7B",
    ),
    "deepseekcoder_33b": ModelSpec(
        key="deepseekcoder_33b",
        model_name="deepseek-ai/deepseek-coder-33b-instruct",
        tp_size=4,
        gpu_count=4,
        note="DeepSeek Coder 33B",
    ),
    "qwen3_0p6b": ModelSpec(
        key="qwen3_0p6b",
        model_name="Qwen/Qwen3-0.6B",
        tp_size=1,
        gpu_count=1,
        note="Qwen3 0.6B",
    ),
    "qwen3_4b": ModelSpec(
        key="qwen3_4b",
        model_name="Qwen/Qwen3-4B",
        tp_size=1,
        gpu_count=1,
        note="Qwen3 4B",
    ),
    "qwen3_8b": ModelSpec(
        key="qwen3_8b",
        model_name="Qwen/Qwen3-8B",
        tp_size=1,
        gpu_count=1,
        note="Qwen3 8B",
    ),
    # "qwen3_30b_a3b": ModelSpec(
    #     key="qwen3_30b_a3b",
    #     model_name="Qwen/Qwen3-30B-A3B",
    #     tp_size=2,
    #     gpu_count=2,
    #     note="Qwen3 30B-A3B",
    # ),
    "qwen3_30b_a3b": ModelSpec(
        key="qwen3_30b_a3b_instruct_2507",
        model_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
        tp_size=4,
        gpu_count=4,
        note="Qwen3 30B-A3B Instruct 2507",
    ),
    "qwen25_0p5b_instruct": ModelSpec(
        key="qwen25_0p5b_instruct",
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        tp_size=1,
        gpu_count=1,
        note="Qwen2.5 0.5B Instruct",
    ),
    "qwen3_4b_instruct_2507": ModelSpec(
        key="qwen3_4b_instruct_2507",
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        tp_size=1,
        gpu_count=1,
        note="Qwen3 4B Instruct 2507",
    ),
    "qwen3_30b_a3b_instruct_2507": ModelSpec(
        key="qwen3_30b_a3b_instruct_2507",
        model_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
        tp_size=4,
        gpu_count=4,
        note="Qwen3 30B-A3B Instruct 2507",
    ),


    "qwen35_0p8b": ModelSpec(
        key="qwen35_0p8b",
        model_name="Qwen/Qwen3.5-0.8B",
        tp_size=1,
        gpu_count=1,
        note="Qwen3.5 0.8B",
    ),

    "qwen35_2b": ModelSpec(
        key="qwen35_2b",
        model_name="Qwen/Qwen3.5-2B",
        tp_size=1,
        gpu_count=1,
        note="Qwen3.5 2B",
    ),

    "qwen35_4b": ModelSpec(
        key="qwen35_4b",
        model_name="Qwen/Qwen3.5-4",
        tp_size=1,
        gpu_count=1,
        note="Qwen3.5 4B",
    ),

    "qwen35_9b": ModelSpec(
        key="qwen35_9b",
        model_name="Qwen/Qwen3.5-4",
        tp_size=1,
        gpu_count=1,
        note="Qwen3.5 9B",
    ),
    "vicuna_68m": ModelSpec(
        key="vicuna_68m",
        model_name="double7/vicuna-68m",
        tp_size=1,
        gpu_count=1,
        note="Vicuna 68M",
    ),
    "vicuna_7b_v13": ModelSpec(
        key="vicuna_7b_v13",
        model_name="lmsys/vicuna-7b-v1.3",
        tp_size=1,
        gpu_count=1,
        note="Vicuna 7B v1.3",
    ),
    "vicuna_13b_v13": ModelSpec(
        key="vicuna_13b_v13",
        model_name="lmsys/vicuna-13b-v1.3",
        tp_size=1,
        gpu_count=1,
        note="Vicuna 13B v1.3",
    ),
}


def make_pair_config(
    draft_key: str,
    target_key: str,
    *,
    pair_id: str | None = None,
    tp_size: int | None = None,
    gpu_count: int | None = None,
    note: str = "",
) -> PairConfig:
    draft = MODEL_SPECS[draft_key]
    target = MODEL_SPECS[target_key]
    return PairConfig(
        pair_id=pair_id or f"{draft.key}_to_{target.key}",
        draft_model=draft.model_name,
        target_model=target.model_name,
        tp_size=tp_size if tp_size is not None else target.tp_size,
        gpu_count=gpu_count if gpu_count is not None else target.gpu_count,
        note=note or f"{draft.note} draft -> {target.note} target",
    )


def build_pairs_from_keys(
    draft_keys: list[str],
    target_keys: list[str],
    *,
    exclude_same: bool = False,
    note_prefix: str = "",
) -> list[PairConfig]:
    pairs: list[PairConfig] = []
    for draft_key in draft_keys:
        for target_key in target_keys:
            if exclude_same and draft_key == target_key:
                continue
            pairs.append(
                make_pair_config(
                    draft_key=draft_key,
                    target_key=target_key,
                    note=f"{note_prefix}{draft_key} -> {target_key}",
                )
            )
    return pairs


def build_chain_pairs(
    model_keys: list[str],
    *,
    note_prefix: str = "",
) -> list[PairConfig]:
    """Ordered chain -> all forward pairs.

    Example:
      [1B, 3B, 8B, 70B]
    becomes:
      1B->3B, 1B->8B, 1B->70B, 3B->8B, 3B->70B, 8B->70B
    """
    pairs: list[PairConfig] = []
    for i, draft_key in enumerate(model_keys):
        for target_key in model_keys[i + 1 :]:
            pairs.append(
                make_pair_config(
                    draft_key=draft_key,
                    target_key=target_key,
                    note=f"{note_prefix}{draft_key} -> {target_key}",
                )
            )
    return pairs


def build_batch_spec_pairs(
    pair_defs: list[BatchSpecPairConfig],
) -> tuple[list[PairConfig], dict[str, str]]:
    pairs: list[PairConfig] = []
    pivot_intermediate_by_pair_id: dict[str, str] = {}
    for spec in pair_defs:
        validate_model_keys([spec.draft_key], arg_name="batch draft key")
        validate_model_keys([spec.target_key], arg_name="batch target key")
        pairs.append(
            make_pair_config(
                spec.draft_key,
                spec.target_key,
                pair_id=spec.pair_id,
                tp_size=spec.tp_size,
                gpu_count=spec.gpu_count,
                note=spec.note,
            )
        )
        if spec.pivot_intermediate_key is not None:
            validate_model_keys(
                [spec.pivot_intermediate_key],
                arg_name="batch pivot intermediate key",
            )
            pivot_intermediate_by_pair_id[spec.pair_id] = MODEL_SPECS[
                spec.pivot_intermediate_key
            ].model_name
    return pairs, pivot_intermediate_by_pair_id


PAIRS: list[PairConfig] = [
    # make_pair_config(
    #     "llama32_1b",
    #     "llama32_3b",
    #     pair_id="llama32_1b_to_llama32_3b",
    #     tp_size=1,
    #     gpu_count=1,
    #     note="Llama small->small",
    # ),
    # make_pair_config(
    #     "llama32_1b",
    #     "llama31_8b",
    #     pair_id="llama32_1b_to_llama31_8b",
    #     tp_size=1,
    #     gpu_count=1,
    #     note="Llama 1B draft -> 8B target",
    # ),
    make_pair_config(
        "llama32_1b",
        "llama33_70b",
        pair_id="llama32_1b_to_llama33_70b",
        tp_size=4,
        gpu_count=4,
        note="Llama 1B draft -> 70B target",
    ),
    # make_pair_config(
    #     "llama32_3b",
    #     "llama31_8b",
    #     pair_id="llama32_3b_to_llama31_8b",
    #     tp_size=1,
    #     gpu_count=1,
    #     note="Llama 3B draft -> 8B target",
    # ),
    # make_pair_config(
    #     "llama32_3b",
    #     "llama33_70b",
    #     pair_id="llama32_3b_to_llama33_70b",
    #     tp_size=4,
    #     gpu_count=4,
    #     note="Llama 3B draft -> 70B target",
    # ),
    # make_pair_config(
    #     "llama31_8b",
    #     "llama33_70b",
    #     pair_id="llama31_8b_to_llama31_70b",
    #     tp_size=4,
    #     gpu_count=4,
    #     note="Llama 8B draft -> 70B target",
    # ),
    # make_pair_config(
    #     "deepseekcoder_1p3b",
    #     "deepseekcoder_6p7b",
    #     pair_id="deepseekcoder_1p3b_to_6p7b",
    #     tp_size=1,
    #     gpu_count=1,
    #     note="DeepSeek Coder 1.3B draft -> 6.7B target",
    # ),
    # make_pair_config(
    #     "deepseekcoder_1p3b",
    #     "deepseekcoder_33b",
    #     pair_id="deepseekcoder_1p3b_to_33b",
    #     tp_size=4,
    #     gpu_count=4,
    #     note="DeepSeek Coder 1.3B draft -> 33B target",
    # ),
    # make_pair_config(
    #     "deepseekcoder_6p7b",
    #     "deepseekcoder_33b",
    #     pair_id="deepseekcoder_6p7b_to_33b",
    #     tp_size=4,
    #     gpu_count=4,
    #     note="DeepSeek Coder 6.7B draft -> 33B target",
    # ),
    # make_pair_config(
    #     "qwen3_0p6b",
    #     "qwen3_4b",
    #     pair_id="qwen3_0p6b_to_qwen3_4b",
    #     tp_size=1,
    #     gpu_count=1,
    #     note="Qwen3 0.6B draft -> 4B target",
    # ),
    # make_pair_config(
    #     "qwen3_0p6b",
    #     "qwen3_8b",
    #     pair_id="qwen3_0p6b_to_qwen3_8b",
    #     tp_size=1,
    #     gpu_count=1,
    #     note="Qwen3 0.6B draft -> 8B target",
    # ),
    # make_pair_config(
    #     "qwen3_0p6b",
    #     "qwen3_30b_a3b",
    #     pair_id="qwen3_0p6b_to_qwen3_30b_a3b",
    #     tp_size=4,
    #     gpu_count=4,
    #     note="Qwen3 0.6B draft -> 30B-A3B target",
    # ),
    # make_pair_config(
    #     "qwen3_4b",
    #     "qwen3_8b",
    #     pair_id="qwen3_4b_to_qwen3_8b",
    #     tp_size=1,
    #     gpu_count=1,
    #     note="Qwen3 4B draft -> 8B target",
    # ),
    # make_pair_config(
    #     "qwen3_4b",
    #     "qwen3_30b_a3b",
    #     pair_id="qwen3_4b_to_qwen3_30b_a3b",
    #     tp_size=4,
    #     gpu_count=4,
    #     note="Qwen3 4B draft -> 30B-A3B target",
    # ),
    # make_pair_config(
    #     "qwen3_8b",
    #     "qwen3_30b_a3b",
    #     pair_id="qwen3_8b_to_qwen3_30b_a3b",
    #     tp_size=4,
    #     gpu_count=4,
    #     note="Qwen3 8B draft -> 30B-A3B target",
    # ),
]

BATCH_SPEC_PAIR_CONFIGS: list[BatchSpecPairConfig] = [
    BatchSpecPairConfig(
        pair_id="llama32_1b_to_llama33_70b",
        draft_key="llama32_1b",
        target_key="llama33_70b",
        tp_size=4,
        gpu_count=4,
        note="Llama 3.2 1B draft -> Llama 3.3 70B target",
        pivot_intermediate_key="llama31_8b",
    ),
    # BatchSpecPairConfig(
    #     pair_id="vicuna_68m_to_vicuna_13b_v13",
    #     draft_key="vicuna_68m",
    #     target_key="vicuna_13b_v13",
    #     tp_size=1,
    #     gpu_count=1,
    #     note="Vicuna 68M draft -> Vicuna 13B v1.3 target",
    #     pivot_intermediate_key="vicuna_7b_v13",
    # ),
]

DATASETS: list[DatasetConfig] = [
    DatasetConfig(name="aime25", max_new_tokens=2048),
    DatasetConfig(name="codeelo", max_new_tokens=2048),
    DatasetConfig(name="gov_report", max_new_tokens=512),
    DatasetConfig(name="qmsum", max_new_tokens=512),
    DatasetConfig(name="alpaca", max_new_tokens=256),
    DatasetConfig(name="gsm8k", max_new_tokens=256),
    DatasetConfig(name="mt_bench", max_new_tokens=256),
    DatasetConfig(name="qa", max_new_tokens=256),
    DatasetConfig(name="humaneval", max_new_tokens=512),
    DatasetConfig(name="sum", max_new_tokens=512),
]

DEFAULT_TEST_PAIR_IDS = frozenset({"llama32_1b_to_llama31_8b"})

DEFAULT_DRAFT_SWEEP_SPEC_TOKENS = [3, 5, 9, 11]
#DEFAULT_VERIFY_SWEEP_SPEC_TOKENS = [3, 5, 9, 11]
DEFAULT_VERIFY_SWEEP_SPEC_TOKENS = [7]
DEFAULT_DRAFT_SWEEP_DRAFT_KEYS = ["llama32_1b", "llama32_3b", "llama31_8b"]
DEFAULT_DRAFT_SWEEP_TARGET_KEYS = ["llama33_70b"]

DEFAULT_VERIFY_SWEEP_MODEL_CHAIN = [
    "llama32_1b",
    "llama32_3b",
    "llama31_8b",
    "llama33_70b",
]


def sanitize_for_path(s: str) -> str:
    s = s.strip().replace("/", "__")
    s = re.sub(r"[^A-Za-z0-9._+@=,]+", "_", s)
    return s.strip("._")


def pair_slug(draft_model: str, target_model: str) -> str:
    return f"{sanitize_for_path(draft_model)}__TO__{sanitize_for_path(target_model)}"


def batch_tag(batch_sizes: str) -> str:
    parts = [p.strip() for p in batch_sizes.split(",") if p.strip()]
    return "b" + "_".join(parts)


def spec_token_tag(num_spec_tokens: int) -> str:
    return f"k{num_spec_tokens}"


def pivot_config_path_tag(topk: int, expansion_pct: float) -> str:
    """Subdir tag for pivot jobs: topk + expansion fraction (e.g. tk5_ep0p2)."""
    return f"tk{topk}_ep{str(expansion_pct).replace('.', 'p')}"


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

    if dataset.name in ("alpaca", "gsm8k", "mt_bench", "qa", "humaneval", "sum"):
        size_b = infer_target_size_b(pair.target_model)
        if size_b is not None and size_b >= 60.0:
            return "08:00:00"
        if size_b is not None and size_b >= 30.0:
            return "05:00:00"
        return "03:00:00"

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
#SBATCH --cpus-per-task={12*gpu_count}
#SBATCH --account=def-pnair_gpu
#SBATCH --qos=def-pnair
#SBATCH --gres=gpu:h100:{gpu_count}
#SBATCH --mem-per-gpu=80G
#SBATCH --time={time_limit}
#SBATCH --output={log_dir / (job_name + ".out")}
#SBATCH --error={log_dir / (job_name + ".err")}

set -euo pipefail

module load python/3.12 cuda/12.9 arrow/21.0.0

REPO_DIR={shquote(str(REPO_DIR))}
VENV_DIR={shquote(str(VENV_DIR))}

cd "${{REPO_DIR}}"
source "${{VENV_DIR}}/bin/activate"

if [[ -f ~/.bashrc ]]; then
  set +u
  source ~/.bashrc
  set -u
fi

unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export HF_TOKEN="${{HF_TOKEN:-}}"
export VLLM_WORKER_MULTIPROC_METHOD="${{VLLM_WORKER_MULTIPROC_METHOD:-spawn}}"

export HF_HOME="${{HOME}}/scratch/.cache"
export TRANSFORMERS_CACHE="${{TRANSFORMERS_CACHE:-/$TRANSFORMERS_CACHE}}"
export HUGGINGFACE_HUB_CACHE="${{HUGGINGFACE_HUB_CACHE:-$HUGGINGFACE_HUB_CACHE}}"
export HF_DATASETS_CACHE="${{HF_DATASETS_CACHE:-$HF_DATASETS_CACHE}}"
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
    debug: bool,
    *,
    test: bool,
    test_samples: int,
    profile_mode: str = "stage_cost",
    method: str = "speculative",
    eagle_model: str | None = None,
    eagle_draft_tp: int | None = None,
    magicdec_method: str = "streaming",
    magicdec_kv_budget: int = 256,
    adaptive_spechive_intermediate_model: str | None = None,
    adaptive_spechive_mode: str = "hierarchical_verification",
    adaptive_spechive_rounds: int = 1,
    pivot_topk_selection: int = 5,
    pivot_expansion_pct: float = 0.2,
    pivot_spechive: bool = False,
    tetris_extra_proposals: int = 0,
    tetris_turn_on_batch_size: int | None = None,
    repo_dir: str | None = None,
    spec_bench_jsonl: str | None = None,
    spec_bench_category: str | None = None,
    mt_bench_dataset_path: str | None = None,
    results_subdir: Path | None = None,
    enforce_eager: bool = False,
) -> str:
    tag = batch_tag(batch_sizes)
    base_root = RESULTS_ROOT / method / tag / dataset.name
    root_for_dataset = base_root / results_subdir if results_subdir is not None else base_root
    slug = pair_slug(pair.draft_model, pair.target_model)

    parts: list[str] = [
        "python -u scripts/run_spec_decode_metrics.py",
        f"--draft-model {shquote(pair.draft_model)}",
        f"--target-models {shquote(pair.target_model)}",
        f"--tp-map {shquote(f'{pair.target_model}={pair.tp_size}')}",
        f"--datasets {dataset.name}",
        f"--repo-dir {shquote(repo_dir if repo_dir is not None else str(REPO_ROOT))}",
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
        f"--spec-decode-profile-mode {profile_mode}",
        "--trust-remote-code",
        "--enable-chunked-prefill",
        f"--method {method}",
    ]

    if method == "eagle3":
        if not eagle_model:
            raise SystemExit("--method=eagle3 requires --eagle-model from generator")
        parts.append(f"--eagle-model {shquote(eagle_model)}")
        parts.append(
            f"--eagle-draft-tp {eagle_draft_tp if eagle_draft_tp is not None else pair.tp_size}"
        )
    elif method == "magicdec":
        parts.append(f"--magicdec-method {shquote(magicdec_method)}")
        parts.append(f"--magicdec-kv-budget {magicdec_kv_budget}")
    elif method == "adaptive_spechive":
        if not adaptive_spechive_intermediate_model:
            raise SystemExit(
                f"--method={method} requires intermediate model in generator "
                "(pass --adaptive-spechive-intermediate-model)."
            )
        parts.append(
            f"--intermediate-model {shquote(adaptive_spechive_intermediate_model)}"
        )
        parts.append(
            f"--adaptive-spechive-mode {shquote(adaptive_spechive_mode)}"
        )
        parts.append(
            f"--round {adaptive_spechive_rounds}"
        )
    elif method == "pivot":
        if pivot_spechive and not adaptive_spechive_intermediate_model:
            raise SystemExit(
                "--method=pivot with --spechive requires intermediate model in "
                "generator (pass --adaptive-spechive-intermediate-model)."
            )
        if adaptive_spechive_intermediate_model:
            parts.append(
                f"--intermediate-model {shquote(adaptive_spechive_intermediate_model)}"
            )
        parts.append(f"--topk_selection {pivot_topk_selection}")
        parts.append(f"--expansion_pct {pivot_expansion_pct}")
        parts.append(f"--round {adaptive_spechive_rounds}")
        if pivot_spechive:
            parts.append("--spechive")
    elif method == "tetris":
        if tetris_extra_proposals:
            parts.append(f"--tetris-extra-proposals {tetris_extra_proposals}")
        if tetris_turn_on_batch_size is not None:
            parts.append(f"--tetris-turn-on-batch-size {tetris_turn_on_batch_size}")

    max_tokens_by_dataset = {
        "aime25": 2048,
        "codeelo": 2048,
        "gov_report": 512,
        "qmsum": 512,
        "spec_bench": 256,
        "alpaca": 256,
        "gsm8k": 256,
        "mt_bench": 256,
        "qa": 256,
        "humaneval": 512,
        "sum": 512,
    }
    if dataset.name not in max_tokens_by_dataset:
        raise SystemExit(f"Unsupported dataset in generator: {dataset.name}")
    max_tokens_by_dataset[dataset.name] = dataset.max_new_tokens
    parts.extend(
        [
            f"--aime-max-new-tokens {max_tokens_by_dataset['aime25']}",
            f"--codeelo-max-new-tokens {max_tokens_by_dataset['codeelo']}",
            f"--gov-report-max-new-tokens {max_tokens_by_dataset['gov_report']}",
            f"--qmsum-max-new-tokens {max_tokens_by_dataset['qmsum']}",
            f"--spec-bench-max-new-tokens {max_tokens_by_dataset['spec_bench']}",
            f"--alpaca-max-new-tokens {max_tokens_by_dataset['alpaca']}",
            f"--gsm8k-max-new-tokens {max_tokens_by_dataset['gsm8k']}",
            f"--mt-bench-max-new-tokens {max_tokens_by_dataset['mt_bench']}",
            f"--qa-max-new-tokens {max_tokens_by_dataset['qa']}",
            f"--humaneval-max-new-tokens {max_tokens_by_dataset['humaneval']}",
            f"--sum-max-new-tokens {max_tokens_by_dataset['sum']}",
        ]
    )

    if dataset.name == "spec_bench":
        if not spec_bench_jsonl:
            raise SystemExit(
                "--datasets spec_bench requires --spec-bench-jsonl in generator."
            )
        parts.append(f"--spec-bench-jsonl {shquote(spec_bench_jsonl)}")
        if spec_bench_category:
            parts.append(f"--spec-bench-category {shquote(spec_bench_category)}")
    if mt_bench_dataset_path:
        parts.append(f"--mt-bench-dataset-path {shquote(mt_bench_dataset_path)}")

    if test:
        ts = test_samples
        parts.append(f"--max-samples-aime25 {ts}")
        parts.append(f"--max-samples-codeelo {ts}")
        parts.append(f"--max-samples-gov-report {ts}")
        parts.append(f"--max-samples-qmsum {ts}")
        parts.append(f"--max-samples-spec-bench {ts}")
        parts.append(f"--max-samples-alpaca {ts}")
        parts.append(f"--max-samples-gsm8k {ts}")
        parts.append(f"--max-samples-mt-bench {ts}")
        parts.append(f"--max-samples-qa {ts}")
        parts.append(f"--max-samples-humaneval {ts}")
        parts.append(f"--max-samples-sum {ts}")

    if verbose:
        parts.append("--verbose")
    if debug:
        parts.append("--debug")
    if enforce_eager:
        parts.append("--enforce-eager")

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
    debug: bool,
    *,
    test: bool,
    test_samples: int,
    profile_mode: str = "stage_cost",
    method: str = "speculative",
    eagle_model: str | None = None,
    eagle_draft_tp: int | None = None,
    magicdec_method: str = "streaming",
    magicdec_kv_budget: int = 256,
    adaptive_spechive_intermediate_model: str | None = None,
    adaptive_spechive_mode: str = "hierarchical_verification",
    adaptive_spechive_rounds: int = 1,
    pivot_topk_selection: int = 5,
    pivot_expansion_pct: float = 0.2,
    pivot_spechive: bool = False,
    tetris_extra_proposals: int = 0,
    tetris_turn_on_batch_size: int | None = None,
    repo_dir: str | None = None,
    spec_bench_jsonl: str | None = None,
    spec_bench_category: str | None = None,
    mt_bench_dataset_path: str | None = None,
    time_limit_override: str | None = None,
    jobs_subdir: Path | None = None,
    results_subdir: Path | None = None,
    job_name_suffix: str = "",
    enforce_eager: bool = False,
) -> str:
    slug = pair_slug(pair.draft_model, pair.target_model)
    tag = batch_tag(batch_sizes)
    suffix = "_test" if test else ""
    suffix += job_name_suffix
    job_name = f"spec_{pair.pair_id}_{dataset.name}{suffix}_{tag}"

    if jobs_subdir is not None:
        pair_subdir = jobs_subdir
    else:
        pair_subdir = Path("test") / tag / pair.pair_id if test else Path(tag) / pair.pair_id

    job_dir = JOBS_ROOT / pair_subdir
    log_dir = LOGS_ROOT / pair_subdir

    result_dataset_root = RESULTS_ROOT / method / tag / dataset.name
    if results_subdir is not None:
        result_dataset_root = result_dataset_root / results_subdir

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
        debug=debug,
        test=test,
        test_samples=test_samples,
        profile_mode=profile_mode,
        method=method,
        eagle_model=eagle_model,
        eagle_draft_tp=eagle_draft_tp,
        magicdec_method=magicdec_method,
        magicdec_kv_budget=magicdec_kv_budget,
        adaptive_spechive_intermediate_model=adaptive_spechive_intermediate_model,
        adaptive_spechive_mode=adaptive_spechive_mode,
        adaptive_spechive_rounds=adaptive_spechive_rounds,
        pivot_topk_selection=pivot_topk_selection,
        pivot_expansion_pct=pivot_expansion_pct,
        pivot_spechive=pivot_spechive,
        tetris_extra_proposals=tetris_extra_proposals,
        tetris_turn_on_batch_size=tetris_turn_on_batch_size,
        repo_dir=repo_dir,
        spec_bench_jsonl=spec_bench_jsonl,
        spec_bench_category=spec_bench_category,
        mt_bench_dataset_path=mt_bench_dataset_path,
        results_subdir=results_subdir,
        enforce_eager=enforce_eager,
    )

    body = f"""
DATASET_NAME={shquote(dataset.name)}
PAIR_ID={shquote(pair.pair_id)}
PAIR_SLUG={shquote(slug)}
TEST_MODE={shquote(str(test))}
NUM_SPEC_TOKENS={num_spec_tokens}
SPECHIVE_DEBUG={shquote("1" if debug else "0")}

JOB_DIR={shquote(str(job_dir))}
RESULT_DATASET_ROOT={shquote(str(result_dataset_root))}
PAIR_RESULT_DIR={shquote(str(pair_result_dir))}
AGGREGATE_CSV={shquote(str(aggregate_csv))}
AGGREGATE_JSONL={shquote(str(aggregate_jsonl))}

mkdir -p "${{JOB_DIR}}" "${{RESULT_DATASET_ROOT}}" "${{PAIR_RESULT_DIR}}"

echo "============================================================"
echo "DATASET: ${{DATASET_NAME}}"
echo "PAIR_ID: ${{PAIR_ID}}"
echo "TEST_MODE: ${{TEST_MODE}}"
echo "DRAFT_MODEL: {pair.draft_model}"
echo "TARGET_MODEL: {pair.target_model}"
echo "TP_SIZE: {pair.tp_size}"
echo "NUM_SPEC_TOKENS: ${{NUM_SPEC_TOKENS}}"
echo "SPECHIVE_DEBUG: ${{SPECHIVE_DEBUG}}"
echo "TIME_LIMIT: {time_limit}"
echo "JOB_DIR: ${{JOB_DIR}}"
echo "RESULT_DATASET_ROOT: ${{RESULT_DATASET_ROOT}}"
echo "PAIR_RESULT_DIR: ${{PAIR_RESULT_DIR}}"
echo "PWD: $(pwd)"
echo "PYTHON: $(which python)"
echo "============================================================"

if [[ "${{SPECHIVE_DEBUG}}" == "1" ]]; then
  export VLLM_SPEC_DIT_DEBUG=1
  export VLLM_SPEC_DIT_DEBUG_SUMMARY=1
  export VLLM_SPEC_SPECHIVE_DEBUG=1
  export VLLM_SPEC_SPECHIVE_DEBUG_SUMMARY=1
fi

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


def parse_int_csv(value: str) -> list[int]:
    out: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return out


def parse_str_csv(value: str) -> list[str]:
    out = [x.strip() for x in value.split(",") if x.strip()]
    if not out:
        raise argparse.ArgumentTypeError("Expected at least one value.")
    return out


def validate_model_keys(keys: list[str], *, arg_name: str) -> None:
    unknown = [k for k in keys if k not in MODEL_SPECS]
    if unknown:
        known = ", ".join(sorted(MODEL_SPECS))
        raise SystemExit(f"{arg_name}: unknown model keys {unknown}. Known: {known}")


def add_submit_script(submit_path: Path, scripts: list[Path]) -> None:
    with open(submit_path, "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("set -euo pipefail\n\n")
        f.write(f'REPO_DIR="{REPO_ROOT}"\n\n')
        for sp in scripts:
            f.write(f'echo "sbatch {sp}"\n')
            f.write(f"sbatch {sp}\n")
    submit_path.chmod(submit_path.stat().st_mode | stat.S_IXUSR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=[], help="Subset of dataset names")
    parser.add_argument(
        "--repo-dir",
        type=str,
        default=str(REPO_ROOT),
        help=(
            "Repository root passed to run_spec_decode_metrics.py for loading "
            "data/<dataset>/question.jsonl."
        ),
    )
    parser.add_argument(
        "--spec-bench-jsonl",
        type=str,
        default=None,
        help=(
            "Path to Spec-Bench question.jsonl. Required when --datasets "
            "includes spec_bench."
        ),
    )
    parser.add_argument(
        "--spec-bench-category",
        type=str,
        default=None,
        help=(
            "Optional Spec-Bench category filter (e.g. alpaca, mt_bench). "
            "Passed through to run_spec_decode_metrics.py."
        ),
    )
    parser.add_argument(
        "--mt-bench-dataset-path",
        type=str,
        default="philschmid/mt-bench",
        help=(
            "Hugging Face dataset path or local JSONL path for MT-Bench. "
            "Passed through to run_spec_decode_metrics.py."
        ),
    )
    parser.add_argument("--pairs", nargs="*", default=[], help="Subset of pair_id values")
    parser.add_argument("--batch-sizes", default="1")

    parser.add_argument(
        "--batch",
        action="store_true",
        help=(
            "Generate speculative decoding jobs for batch sizes "
            "(1,4,16,64,256,512) with fixed time limits."
        ),
    )
    parser.add_argument(
        "--length",
        action="store_true",
        help=(
            "Like --batch but batch size fixed at 256 and num_spec_tokens "
            "swept over 5, 7, 9, 11 for each method."
        ),
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help=(
            "Pivot-method jobs only: topk_selection in {2, 5} and "
            "expansion_pct in {0.1, 0.2} (full grid), num_spec_tokens in "
            "{3, 7, 11}, same batch-size sweep as --batch."
        ),
    )
    parser.add_argument(
        "--draft",
        action="store_true",
        help=(
            "Generate num_spec_tokens sweep jobs while varying draft model "
            "and keeping target fixed."
        ),
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "Generate num_spec_tokens sweep jobs while varying target model "
            "along an ordered model chain."
        ),
    )

    parser.add_argument(
        "--eager",
        action="store_true",
        help=(
            "Pass --enforce-eager to run_spec_decode_metrics.py in generated "
            "Slurm scripts (disable CUDA graph / compile for easier debugging)."
        ),
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--warmup-max-tokens", type=int, default=1024)
    parser.add_argument(
        "--magicdec-method",
        default="streaming",
        choices=["streaming"],
        help="MagicDec method to pass when method=magicdec (default: streaming).",
    )
    parser.add_argument(
        "--magicdec-kv-budget",
        type=int,
        default=256,
        help="MagicDec KV budget tokens when method=magicdec (default: 256).",
    )
    parser.add_argument(
        "--adaptive-spechive-intermediate-model",
        type=str,
        default=None,
        help=(
            "Intermediate verifier model id to pass when generating "
            "method=adaptive_spechive jobs."
        ),
    )
    parser.add_argument(
        "--adaptive-spechive-mode",
        type=str,
        default="hierarchical_verification",
        choices=["draft_target", "inter_verification", "hierarchical_verification"],
        help="adaptive_spechive mode to pass to run_spec_decode_metrics.py.",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=1,
        help="Number of adaptive_spechive hierarchical verification rounds.",
    )
    parser.add_argument(
        "--topk-selection",
        type=int,
        default=5,
        choices=[2, 5],
        help="When --spec-method=pivot: first-token top-k expansion width.",
    )
    parser.add_argument(
        "--expansion-pct",
        type=float,
        default=0.2,
        help="When --spec-method=pivot: fraction of low-confidence requests to expand.",
    )
    parser.add_argument(
        "--pivot-spechive",
        action="store_true",
        help="When --spec-method=pivot: enable staged D=>I rounds before D=>T verification.",
    )
    parser.add_argument(
        "--spec-method",
        type=str,
        default="speculative",
        choices=["speculative", "adaptive_spechive", "pivot", "tetris"],
        help=(
            "Method used for speculative jobs in this generator. "
            "Use adaptive_spechive for staged D/I/T runs, pivot for "
            "intermediate-pivot plus draft-tail, or tetris for TETRIS "
            "optimal draft token selection (ACL 2025)."
        ),
    )
    parser.add_argument(
        "--tetris-extra-proposals",
        type=int,
        default=2,
        help=(
            "When --spec-method=tetris: extra draft tokens beyond base_k. "
            "base_k = num_spec_tokens; drafter auto-generates "
            "K = base_k + extra; capacity = base_k * batch_size (default: 2)."
        ),
    )
    parser.add_argument(
        "--tetris-turn-on-batch-size",
        type=int,
        default=None,
        help=(
            "When --spec-method=tetris: minimum batch size before TETRIS "
            "activates; smaller batches use standard spec-decode (default: always on)."
        ),
    )
    parser.add_argument(
        "--profile-mode",
        type=str,
        default="stage_cost",
        choices=["disabled", "stage_cost", "shape_memory", "kernel_breakdown", "all"],
        help=(
            "Speculative-decoding profiler mode for generated jobs. "
            "Default: stage_cost (per-stage wall-time cost breakdown)."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Enable DIT runtime debug validation in generated jobs "
            "(exports VLLM_SPEC_DIT_DEBUG=1 and passes --debug)."
        ),
    )
    parser.add_argument(
        "--spechive-debug",
        action="store_true",
        help=(
            "Enable spechive runtime debug checks in generated jobs "
            "(exports VLLM_SPEC_SPECHIVE_DEBUG=1 and passes --debug)."
        ),
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help=(
            "Write smoke-test jobs under jobs/spec_decode/test/<batch_tag>/<pair_id>/. "
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

    parser.add_argument(
        "--draft-num-spec-tokens",
        type=parse_int_csv,
        default=DEFAULT_DRAFT_SWEEP_SPEC_TOKENS,
        help="Comma-separated num_spec_tokens for --draft (default: 3,5,9,11).",
    )
    parser.add_argument(
        "--verify-num-spec-tokens",
        type=parse_int_csv,
        default=DEFAULT_VERIFY_SWEEP_SPEC_TOKENS,
        help="Comma-separated num_spec_tokens for --verify (default: 3,5,9,11).",
    )

    parser.add_argument(
        "--draft-draft-keys",
        type=parse_str_csv,
        default=DEFAULT_DRAFT_SWEEP_DRAFT_KEYS,
        help="Comma-separated draft model keys for --draft.",
    )
    parser.add_argument(
        "--draft-target-keys",
        type=parse_str_csv,
        default=DEFAULT_DRAFT_SWEEP_TARGET_KEYS,
        help="Comma-separated target model keys for --draft.",
    )
    parser.add_argument(
        "--verify-model-chain",
        type=parse_str_csv,
        default=DEFAULT_VERIFY_SWEEP_MODEL_CHAIN,
        help=(
            "Comma-separated ordered model keys for --verify. "
            "Forward pairs are generated automatically."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    debug_enabled = args.debug or args.spechive_debug
    spechive_variants: list[tuple[str | None, int, str]] = []
    if args.spec_method not in ("adaptive_spechive", "pivot"):
        spechive_variants = [(None, args.round, "")]
    elif args.spec_method == "adaptive_spechive" and debug_enabled:
        # Debug preset: run two intermediate verifiers with 3 rounds.
        spechive_variants = [
            ("meta-llama/Llama-3.2-3B-Instruct", 3, "im_llama32_3b_r3"),
            ("meta-llama/Meta-Llama-3.1-8B-Instruct", 3, "im_llama31_8b_r3"),
        ]
    elif args.spec_method == "adaptive_spechive":
        if not args.adaptive_spechive_intermediate_model:
            raise SystemExit(
                f"--spec-method={args.spec_method} requires "
                "--adaptive-spechive-intermediate-model"
            )
        spechive_variants = [
            (args.adaptive_spechive_intermediate_model, args.round, "")
        ]
    else:
        # --spec-method=pivot
        if args.pivot_spechive and not args.adaptive_spechive_intermediate_model:
            raise SystemExit(
                "--spec-method=pivot with --pivot-spechive requires "
                "--adaptive-spechive-intermediate-model"
            )
        spechive_variants = [
            (args.adaptive_spechive_intermediate_model, args.round, "")
        ]

    selected_modes = [
        args.batch,
        args.test,
        args.draft,
        args.verify,
        args.length,
        args.ablation,
    ]
    if sum(1 for x in selected_modes if x) > 1:
        raise SystemExit(
            "Use only one of --batch, --length, --ablation, --test, --draft, or --verify."
        )

    datasets = filter_by_attr(DATASETS, set(args.datasets), "name")

    if args.batch or args.length or args.ablation:
        if not datasets:
            raise SystemExit(
                "With --batch/--length/--ablation: pass at least one --datasets value. "
                f"Known: {[d.name for d in DATASETS]}"
            )

        if args.batch:
            batch_sizes_list = [1, 4, 16, 64, 256]
            num_spec_values = [3]
            pivot_ablation_pairs: list[tuple[int, float]] = [
                (args.topk_selection, args.expansion_pct)
            ]
            pivot_only = False
            scaling_mode = "batch"
        elif args.length:
            batch_sizes_list = [256]
            num_spec_values = [7, 11]
            pivot_ablation_pairs = [(args.topk_selection, args.expansion_pct)]
            pivot_only = False
            scaling_mode = "length"
        else:
            # Same batch sweep as --batch in this script; pivot grid + k sweep.
            batch_sizes_list = [1, 4, 16, 64, 256]
            num_spec_values = [3, 7, 11]
            pivot_ablation_pairs = [(2, 0.1), (2, 0.2), (5, 0.1), (5, 0.2)]
            pivot_only = True
            scaling_mode = "ablation"

        def time_limit_for_batch(bs: int) -> str:
            if bs == 1:
                return "05:00:00"
            if bs == 4:
                return "03:00:00"
            if bs == 16:
                return "02:00:00"
            if bs == 64:
                return "01:00:00"
            if bs == 256:
                return "01:00:00"
            if bs in (512, 1024):
                return "00:40:00"
            raise ValueError(f"Unsupported batch size: {bs}")

        llama33_70b = "meta-llama/Llama-3.3-70B-Instruct"
        llama31_70b_old = "meta-llama/Meta-Llama-3.1-70B-Instruct"
        qwen30b_a3b = "Qwen/Qwen3-30B-A3B-Instruct-2507"

        eagle_llama33_speculator = os.environ.get(
            "EAGLE3_LLAMA33_70B_SPECULATOR",
            "RedHatAI/Llama-3.3-70B-Instruct-speculator.eagle3",
        )
        eagle_qwen30b_a3b_speculator = os.environ.get(
            "EAGLE3_QWEN30B_A3B_SPECULATOR",
            "RedHatAI/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3",
        )

        ar_pairs: list[PairConfig] = [
            PairConfig(
                pair_id="ar_llama33_70b",
                draft_model=llama33_70b,
                target_model=llama33_70b,
                tp_size=4,
                gpu_count=4,
                note="AR only (no speculative decoding)",
            ),
            # PairConfig(
            #     pair_id="ar_qwen30b_a3b",
            #     draft_model=qwen30b_a3b,
            #     target_model=qwen30b_a3b,
            #     tp_size=4,
            #     gpu_count=4,
            #     note="AR only (no speculative decoding)",
            # ),
            # PairConfig(
            #     pair_id="ar_deepseekcoder_33b",
            #     draft_model="deepseek-ai/deepseek-coder-33b-instruct",
            #     target_model="deepseek-ai/deepseek-coder-33b-instruct",
            #     tp_size=4,
            #     gpu_count=4,
            #     note="AR only (no speculative decoding)",
            # ),
        ]

        eagle_pairs: list[PairConfig] = [
            PairConfig(
                pair_id="eagle3_llama33_70b",
                draft_model=llama33_70b,
                target_model=llama33_70b,
                tp_size=4,
                gpu_count=4,
                note="EAGLE3 (eagle3 method)",
            ),
            # PairConfig(
            #     pair_id="eagle3_qwen30b_a3b",
            #     draft_model=qwen30b_a3b,
            #     target_model=qwen30b_a3b,
            #     tp_size=4,
            #     gpu_count=4,
            #     note="EAGLE3 (eagle3 method)",
            # ),
        ]

        speculative_pairs: list[PairConfig] = []
        for pair in PAIRS:
            if pair.target_model == llama31_70b_old:
                speculative_pairs.append(
                    PairConfig(
                        pair_id=pair.pair_id.replace("llama31_70b", "llama33_70b"),
                        draft_model=pair.draft_model,
                        target_model=llama33_70b,
                        tp_size=4,
                        gpu_count=4,
                        note=pair.note,
                    )
                )
            else:
                speculative_pairs.append(pair)

        speculative_pairs.extend(
            [   
                # Qwen 3.5
                # make_pair_config(
                #     "qwen35_0p8b",
                #     "qwen3_30b_a3b",
                #     pair_id="qwen35_0p8b_to_qwen3_30b_a3b",
                #     tp_size=4,
                #     gpu_count=4,
                #     note="Qwen3.5 0.8B draft -> 30B-A3B target",
                # ),

                # make_pair_config(
                #     "qwen35_2b",
                #     "qwen3_30b_a3b",
                #     pair_id="qwen35_2b_to_qwen3_30b_a3b",
                #     tp_size=4,
                #     gpu_count=4,
                #     note="Qwen3.5 2B draft -> 30B-A3B target",
                # ),
                
                # make_pair_config(
                #     "qwen35_4b",
                #     "qwen3_30b_a3b",
                #     pair_id="qwen35_4b_to_qwen3_30b_a3b",
                #     tp_size=4,
                #     gpu_count=4,
                #     note="Qwen3.5 4B draft -> 30B-A3B target",
                # ),
                
                # make_pair_config(
                #     "qwen35_9b",
                #     "qwen3_30b_a3b",
                #     pair_id="qwen35_9b_to_qwen3_30b_a3b",
                #     tp_size=4,
                #     gpu_count=4,
                #     note="Qwen3.5 9B draft -> 30B-A3B target",
                # ),
                
                # make_pair_config(
                #     "qwen25_0p5b_instruct",
                #     "qwen3_30b_a3b",
                #     pair_id="qwen25_0p5b_to_qwen3_30b_a3b",
                #     tp_size=4,
                #     gpu_count=4,
                #     note="Qwen2.5 0.5B draft -> 30B-A3B target",
                # ),

                

                # make_pair_config(
                #     "qwen3_4b_instruct_2507",
                #     "qwen3_30b_a3b",
                #     pair_id="qwen3_4b-instruct_to_qwen3_30b_a3b",
                #     tp_size=4,
                #     gpu_count=4,
                #     note="Qwen3 4B-Inst draft -> 30B-A3B target",
                # ),
                
                # make_pair_config(
                #     "qwen3_0p6b",
                #     "qwen3_30b_a3b",
                #     pair_id="qwen3_0p6b_to_qwen3_30b_a3b",
                #     tp_size=4,
                #     gpu_count=4,
                #     note="Qwen3 0.6B draft -> 30B-A3B target",
                # ),

                # make_pair_config(
                #     "qwen3_4b",
                #     "qwen3_30b_a3b",
                #     pair_id="qwen3_4b_to_qwen3_30b_a3b",
                #     tp_size=4,
                #     gpu_count=4,
                #     note="Qwen3 4B draft -> 30B-A3B target",
                # ),

                # PairConfig(
                #     pair_id="qwen25_0p5b_to_qwen3_4b_instruct_2507",
                #     draft_model="Qwen/Qwen2.5-0.5B-Instruct",
                #     target_model="Qwen/Qwen3-4B-Instruct-2507",
                #     tp_size=1,
                #     gpu_count=1,
                #     note="Qwen2.5 0.5B draft -> Qwen3 4B Instruct 2507",
                # ),
                # PairConfig(
                #     pair_id="qwen25_0p5b_to_qwen3_30b_a3b_instruct_2507",
                #     draft_model="Qwen/Qwen2.5-0.5B-Instruct",
                #     target_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
                #     tp_size=2,
                #     gpu_count=2,
                #     note="Qwen2.5 0.5B draft -> Qwen3 30B-A3B Instruct 2507",
                # ),
                # PairConfig(
                #     pair_id="qwen3_0p6b_to_qwen3_4b_instruct_2507",
                #     draft_model="Qwen/Qwen3-0.6B",
                #     target_model="Qwen/Qwen3-4B-Instruct-2507",
                #     tp_size=1,
                #     gpu_count=1,
                #     note="Qwen3 0.6B draft -> Qwen3 4B Instruct 2507",
                # ),
                # PairConfig(
                #     pair_id="qwen3_0p6b_to_qwen3_30b_a3b_instruct_2507",
                #     draft_model="Qwen/Qwen3-0.6B",
                #     target_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
                #     tp_size=4,
                #     gpu_count=4,
                #     note="Qwen3 0.6B draft -> Qwen3 30B-A3B Instruct 2507",
                # ),
            ]
        )
        configured_pairs, configured_pivot_intermediate = build_batch_spec_pairs(
            BATCH_SPEC_PAIR_CONFIGS
        )
        speculative_by_id: dict[str, PairConfig] = {
            pair.pair_id: pair for pair in speculative_pairs
        }
        for pair in configured_pairs:
            speculative_by_id[pair.pair_id] = pair
        speculative_pairs = list(speculative_by_id.values())
        pivot_intermediate_by_pair_id = configured_pivot_intermediate

        test = False
        test_samples = 5
        warmup_iters = args.warmup_iters
        warmup_max_tokens = args.warmup_max_tokens
        max_model_len = args.max_model_len

        written_scripts: list[Path] = []

        def _write_one(
            *,
            pair: PairConfig,
            dataset: DatasetConfig,
            bs: int,
            num_spec_tokens: int,
            method: str,
            eagle_model: str | None,
            eagle_draft_tp: int | None = None,
            magicdec_method: str = "streaming",
            magicdec_kv_budget: int = 256,
            pivot_spechive: bool | None = None,
            adaptive_spechive_intermediate_model: str | None = None,
            tetris_extra_proposals: int = 0,
            tetris_turn_on_batch_size: int | None = None,
            time_limit_override: str,
            pivot_topk_sel: int | None = None,
            pivot_exp_pct: float | None = None,
            pivot_path_component: str = "",
        ) -> None:
            batch_sizes_str = str(bs)
            ktag = spec_token_tag(num_spec_tokens)
            topk_eff = (
                pivot_topk_sel if pivot_topk_sel is not None else args.topk_selection
            )
            exp_pct_eff = (
                pivot_exp_pct if pivot_exp_pct is not None else args.expansion_pct
            )
            pivot_mode_tag = ""
            if method == "pivot" and pivot_spechive is not None:
                pivot_mode_tag = "pivot_spechive" if pivot_spechive else "pivot"

            method_dir = Path(method) / pivot_mode_tag if pivot_mode_tag else Path(method)
            base_batch_dir = method_dir / f"b{bs}" / ktag
            if pivot_path_component:
                base_batch_dir = base_batch_dir / pivot_path_component
            base_batch_dir = base_batch_dir / pair.pair_id
            base_result_subdir = Path(ktag)
            if pivot_path_component:
                base_result_subdir = base_result_subdir / pivot_path_component
            if pivot_mode_tag:
                base_result_subdir = base_result_subdir / pivot_mode_tag
            if method == "adaptive_spechive":
                variants = spechive_variants
            elif method == "pivot" and pivot_spechive:
                im_model = (
                    adaptive_spechive_intermediate_model
                    if adaptive_spechive_intermediate_model is not None
                    else args.adaptive_spechive_intermediate_model
                )
                variants = [
                    (im_model, args.round, "")
                ]
            else:
                variants = [(None, args.round, "")]
            for im_model, rounds, variant_tag in variants:
                batch_dir = (
                    base_batch_dir / variant_tag if variant_tag else base_batch_dir
                )
                result_subdir = (
                    base_result_subdir / variant_tag
                    if variant_tag
                    else base_result_subdir
                )
                script_path = JOBS_ROOT / batch_dir / f"{dataset.name}.slurm"
                suffix_parts = [ktag]
                if pivot_path_component:
                    suffix_parts.append(pivot_path_component)
                if pivot_mode_tag:
                    suffix_parts.append(pivot_mode_tag)
                if variant_tag:
                    suffix_parts.append(variant_tag)
                suffix = "_" + "_".join(suffix_parts)

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
                    num_spec_tokens=num_spec_tokens,
                    verbose=args.verbose,
                    debug=debug_enabled,
                    test=test,
                    test_samples=test_samples,
                    profile_mode=args.profile_mode,
                    method=method,
                    eagle_model=eagle_model,
                    eagle_draft_tp=eagle_draft_tp,
                    magicdec_method=magicdec_method,
                    magicdec_kv_budget=magicdec_kv_budget,
                    adaptive_spechive_intermediate_model=im_model,
                    adaptive_spechive_mode=args.adaptive_spechive_mode,
                    adaptive_spechive_rounds=rounds,
                    pivot_topk_selection=topk_eff,
                    pivot_expansion_pct=exp_pct_eff,
                    pivot_spechive=(
                        args.pivot_spechive
                        if pivot_spechive is None
                        else pivot_spechive
                    ),
                    tetris_extra_proposals=tetris_extra_proposals,
                    tetris_turn_on_batch_size=tetris_turn_on_batch_size,
                    repo_dir=args.repo_dir,
                    spec_bench_jsonl=args.spec_bench_jsonl,
                    spec_bench_category=args.spec_bench_category,
                    mt_bench_dataset_path=args.mt_bench_dataset_path,
                    time_limit_override=time_limit_override,
                    jobs_subdir=batch_dir,
                    results_subdir=result_subdir,
                    job_name_suffix=suffix,
                    enforce_eager=args.eager,
                )
                write_job_script(script_path, script_text)
                written_scripts.append(script_path)

        def resolve_pivot_intermediate(pair_id: str) -> str | None:
            pair_specific = pivot_intermediate_by_pair_id.get(pair_id)
            if pair_specific is not None:
                return pair_specific
            return args.adaptive_spechive_intermediate_model

        include_pivot_spechive_batch = any(
            resolve_pivot_intermediate(pair.pair_id) is not None
            for pair in speculative_pairs
        )
        if (
            args.spec_method == "pivot"
            and args.pivot_spechive
            and not include_pivot_spechive_batch
        ):
            raise SystemExit(
                f"--{scaling_mode} with --spec-method=pivot --pivot-spechive requires "
                "--adaptive-spechive-intermediate-model"
            )

        for num_spec in num_spec_values:
            for bs in batch_sizes_list:
                tl = time_limit_for_batch(bs)
                for dataset in datasets:
                    if not pivot_only and args.spec_method != "pivot":
                        for pair in speculative_pairs:
                            _write_one(
                                pair=pair,
                                dataset=dataset,
                                bs=bs,
                                num_spec_tokens=num_spec,
                                method=args.spec_method,
                                eagle_model=None,
                                time_limit_override=tl,
                            )

                    if not pivot_only:
                        # MagicDec
                        for pair in speculative_pairs:
                            _write_one(
                                pair=pair,
                                dataset=dataset,
                                bs=bs,
                                num_spec_tokens=num_spec,
                                method="magicdec",
                                eagle_model=None,
                                magicdec_method=args.magicdec_method,
                                magicdec_kv_budget=args.magicdec_kv_budget,
                                time_limit_override=tl,
                            )

                        # TETRIS
                        for pair in speculative_pairs:
                            _write_one(
                                pair=pair,
                                dataset=dataset,
                                bs=bs,
                                num_spec_tokens=num_spec,
                                method="tetris",
                                eagle_model=None,
                                tetris_extra_proposals=args.tetris_extra_proposals,
                                tetris_turn_on_batch_size=args.tetris_turn_on_batch_size,
                                time_limit_override=tl,
                            )

                    for pivot_topk, pivot_exp in pivot_ablation_pairs:
                        pivot_pc = pivot_config_path_tag(pivot_topk, pivot_exp)
                        # PIVOT (always)
                        for pair in speculative_pairs:
                            _write_one(
                                pair=pair,
                                dataset=dataset,
                                bs=bs,
                                num_spec_tokens=num_spec,
                                method="pivot",
                                eagle_model=None,
                                pivot_spechive=False,
                                time_limit_override=tl,
                                pivot_topk_sel=pivot_topk,
                                pivot_exp_pct=pivot_exp,
                                pivot_path_component=pivot_pc,
                            )

                        # PIVOT+SpecHive (only when intermediate model is available)
                        if include_pivot_spechive_batch:
                            for pair in speculative_pairs:
                                pivot_intermediate = resolve_pivot_intermediate(
                                    pair.pair_id
                                )
                                if pivot_intermediate is None:
                                    continue
                                _write_one(
                                    pair=pair,
                                    dataset=dataset,
                                    bs=bs,
                                    num_spec_tokens=num_spec,
                                    method="pivot",
                                    eagle_model=None,
                                    pivot_spechive=True,
                                    adaptive_spechive_intermediate_model=pivot_intermediate,
                                    time_limit_override=tl,
                                    pivot_topk_sel=pivot_topk,
                                    pivot_exp_pct=pivot_exp,
                                    pivot_path_component=pivot_pc,
                                )

                # Enable these if you want AR and EAGLE3 jobs as well.
                # for pair in ar_pairs:
                #     _write_one(
                #         pair=pair,
                #         dataset=dataset,
                #         bs=bs,
                #         method="ar",
                #         eagle_model=None,
                #         time_limit_override=tl,
                #     )

                if not pivot_only:
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
                            num_spec_tokens=num_spec,
                            method="eagle3",
                            eagle_model=eagle_model,
                            eagle_draft_tp=1,
                            time_limit_override=tl,
                        )

        num_written = len(written_scripts)
        print(f"[{scaling_mode}] Generated {num_written} job scripts.")

        submit_scripts = {
            "batch": REPO_ROOT / "scripts" / "submit_spec_decode_batch_scaling.sh",
            "length": REPO_ROOT / "scripts" / "submit_spec_decode_length_sweep.sh",
            "ablation": REPO_ROOT / "scripts" / "submit_spec_decode_pivot_ablation.sh",
        }
        submit_path = submit_scripts[scaling_mode]
        add_submit_script(submit_path, written_scripts)
        print(f"[{scaling_mode}] Submit script: {submit_path}")
        return

    if args.draft:
        if not datasets:
            raise SystemExit(
                "With --draft: please pass at least one --datasets value. "
                f"Known: {[d.name for d in DATASETS]}"
            )


        validate_model_keys(args.draft_draft_keys, arg_name="--draft-draft-keys")
        validate_model_keys(args.draft_target_keys, arg_name="--draft-target-keys")

        pairs = build_pairs_from_keys(
            draft_keys=args.draft_draft_keys,
            target_keys=args.draft_target_keys,
            exclude_same=True,
            note_prefix="draft_sweep ",
        )

        written_scripts: list[Path] = []
        batch_sizes = args.batch_sizes
        tag = batch_tag(batch_sizes)

        for pair in pairs:
            for num_spec in args.draft_num_spec_tokens:
                ktag = spec_token_tag(num_spec)
                for dataset in datasets:
                    base_pair_subdir = Path("draft") / tag / ktag / pair.pair_id
                    base_results_subdir = Path("draft") / ktag
                    for im_model, rounds, variant_tag in spechive_variants:
                        pair_subdir = (
                            base_pair_subdir / variant_tag
                            if variant_tag
                            else base_pair_subdir
                        )
                        results_subdir = (
                            base_results_subdir / variant_tag
                            if variant_tag
                            else base_results_subdir
                        )
                        script_path = JOBS_ROOT / pair_subdir / f"{dataset.name}.slurm"
                        suffix = f"_{ktag}" + (f"_{variant_tag}" if variant_tag else "")

                        script_text = render_job_script(
                            pair=pair,
                            dataset=dataset,
                            batch_sizes=batch_sizes,
                            gpu_mem_util=args.gpu_memory_utilization,
                            max_model_len=args.max_model_len,
                            dtype=args.dtype,
                            seed=args.seed,
                            warmup_iters=args.warmup_iters,
                            warmup_max_tokens=args.warmup_max_tokens,
                            num_spec_tokens=num_spec,
                            verbose=args.verbose,
                            debug=debug_enabled,
                            test=False,
                            test_samples=5,
                            profile_mode=args.profile_mode,
                            method=args.spec_method,
                            adaptive_spechive_intermediate_model=im_model,
                            adaptive_spechive_mode=args.adaptive_spechive_mode,
                            adaptive_spechive_rounds=rounds,
                            pivot_topk_selection=args.topk_selection,
                            pivot_expansion_pct=args.expansion_pct,
                            pivot_spechive=args.pivot_spechive,
                            tetris_extra_proposals=args.tetris_extra_proposals,
                            tetris_turn_on_batch_size=args.tetris_turn_on_batch_size,
                            repo_dir=args.repo_dir,
                            spec_bench_jsonl=args.spec_bench_jsonl,
                            spec_bench_category=args.spec_bench_category,
                            mt_bench_dataset_path=args.mt_bench_dataset_path,
                            jobs_subdir=pair_subdir,
                            results_subdir=results_subdir,
                            job_name_suffix=suffix,
                            time_limit_override="1:00:00",
                            enforce_eager=args.eager,
                        )
                        write_job_script(script_path, script_text)
                        written_scripts.append(script_path)
                        print(f"Wrote {script_path}")

        print(f"[draft] Generated {len(written_scripts)} job scripts.")
        submit_path = REPO_ROOT / "scripts" / "submit_spec_decode_draft_sweep.sh"
        add_submit_script(submit_path, written_scripts)
        print(f"[draft] Submit script: {submit_path}")
        return

    if args.verify:
        if not datasets:
            raise SystemExit(
                "With --verify: please pass at least one --datasets value. "
                f"Known: {[d.name for d in DATASETS]}"
            )

        validate_model_keys(args.verify_model_chain, arg_name="--verify-model-chain")

        pairs = build_chain_pairs(
            args.verify_model_chain,
            note_prefix="verify_sweep ",
        )

        written_scripts: list[Path] = []
        batch_sizes = args.batch_sizes
        tag = batch_tag(batch_sizes)
        for pair in pairs:
            for num_spec in args.verify_num_spec_tokens:
                ktag = spec_token_tag(num_spec)
                for dataset in datasets:
                    base_pair_subdir = Path("verify") / tag / ktag / pair.pair_id
                    base_results_subdir = Path("verify") / ktag
                    for im_model, rounds, variant_tag in spechive_variants:
                        pair_subdir = (
                            base_pair_subdir / variant_tag
                            if variant_tag
                            else base_pair_subdir
                        )
                        results_subdir = (
                            base_results_subdir / variant_tag
                            if variant_tag
                            else base_results_subdir
                        )
                        script_path = JOBS_ROOT / pair_subdir / f"{dataset.name}.slurm"
                        suffix = f"_{ktag}" + (f"_{variant_tag}" if variant_tag else "")

                        script_text = render_job_script(
                            pair=pair,
                            dataset=dataset,
                            batch_sizes=batch_sizes,
                            gpu_mem_util=args.gpu_memory_utilization,
                            max_model_len=args.max_model_len,
                            dtype=args.dtype,
                            seed=args.seed,
                            warmup_iters=args.warmup_iters,
                            warmup_max_tokens=args.warmup_max_tokens,
                            num_spec_tokens=num_spec,
                            verbose=args.verbose,
                            debug=debug_enabled,
                            test=False,
                            test_samples=5,
                            profile_mode=args.profile_mode,
                            method=args.spec_method,
                            adaptive_spechive_intermediate_model=im_model,
                            adaptive_spechive_mode=args.adaptive_spechive_mode,
                            adaptive_spechive_rounds=rounds,
                            pivot_topk_selection=args.topk_selection,
                            pivot_expansion_pct=args.expansion_pct,
                            pivot_spechive=args.pivot_spechive,
                            tetris_extra_proposals=args.tetris_extra_proposals,
                            tetris_turn_on_batch_size=args.tetris_turn_on_batch_size,
                            repo_dir=args.repo_dir,
                            spec_bench_jsonl=args.spec_bench_jsonl,
                            spec_bench_category=args.spec_bench_category,
                            mt_bench_dataset_path=args.mt_bench_dataset_path,
                            jobs_subdir=pair_subdir,
                            results_subdir=results_subdir,
                            job_name_suffix=suffix,
                            time_limit_override="1:00:00",
                            enforce_eager=args.eager,
                        )
                        write_job_script(script_path, script_text)
                        written_scripts.append(script_path)
                        print(f"Wrote {script_path}")

        print(f"[verify] Generated {len(written_scripts)} job scripts.")
        submit_path = REPO_ROOT / "scripts" / "submit_spec_decode_verify_sweep.sh"
        add_submit_script(submit_path, written_scripts)
        print(f"[verify] Submit script: {submit_path}")
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
        num_spec = 3
        max_model_len = args.max_model_len

    tag = batch_tag(batch_sizes)
    ktag = spec_token_tag(num_spec)

    num_written = 0
    for pair in pairs:
        for dataset in datasets:
            if test:
                base_pair_subdir = Path("test") / tag / ktag / pair.pair_id
                base_results_subdir = Path("test") / ktag
            else:
                base_pair_subdir = Path(tag) / ktag / pair.pair_id
                base_results_subdir = Path(ktag)

            for im_model, rounds, variant_tag in spechive_variants:
                pair_subdir = (
                    base_pair_subdir / variant_tag
                    if variant_tag
                    else base_pair_subdir
                )
                results_subdir = (
                    base_results_subdir / variant_tag
                    if variant_tag
                    else base_results_subdir
                )
                script_path = JOBS_ROOT / pair_subdir / f"{dataset.name}.slurm"
                suffix = f"_{ktag}" + (f"_{variant_tag}" if variant_tag else "")
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
                    debug=debug_enabled,
                    test=test,
                    test_samples=test_samples,
                    profile_mode=args.profile_mode,
                    method=args.spec_method,
                    adaptive_spechive_intermediate_model=im_model,
                    adaptive_spechive_mode=args.adaptive_spechive_mode,
                    adaptive_spechive_rounds=rounds,
                    pivot_topk_selection=args.topk_selection,
                    pivot_expansion_pct=args.expansion_pct,
                    pivot_spechive=args.pivot_spechive,
                    tetris_extra_proposals=args.tetris_extra_proposals,
                    tetris_turn_on_batch_size=args.tetris_turn_on_batch_size,
                    repo_dir=args.repo_dir,
                    spec_bench_jsonl=args.spec_bench_jsonl,
                    spec_bench_category=args.spec_bench_category,
                    mt_bench_dataset_path=args.mt_bench_dataset_path,
                    jobs_subdir=pair_subdir,
                    results_subdir=results_subdir,
                    job_name_suffix=suffix,
                    enforce_eager=args.eager,
                )
                write_job_script(script_path, script_text)
                num_written += 1
                print(f"Wrote {script_path}")

    mode = "test" if test else "prod"
    print(f"Generated {num_written} job scripts ({mode}).")


if __name__ == "__main__":
    main()