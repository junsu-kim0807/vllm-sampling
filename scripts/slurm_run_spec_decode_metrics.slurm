#!/bin/bash
#SBATCH --job-name=spec_metrics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --account=rrg-pnair_gpu
#SBATCH --qos=rrg-pnair
#SBATCH --gres=gpu:h100:1
#SBATCH --mem-per-gpu=80G
#SBATCH --time=6:00:00
#SBATCH --output=logs/spec_decode_metrics_%x_%j.out
#SBATCH --error=logs/spec_decode_metrics_%x_%j.err
# Override when submitting: --time, --gres, --job-name, --export=ALL,TARGET_CFG=4b|8b|30b

set -euo pipefail

# -----------------------------
# User config (TARGET_CFG set by submit script: 4b, 8b, or 30b)
# -----------------------------
module load python/3.12 cuda/12.9 arrow/21.0.0

REPO_DIR="/project/6045356/junsu/kv_cache/vllm-sampling"
VENV_DIR="/project/6045356/junsu/kv_cache/.venv"

DRAFT_MODEL="Qwen/Qwen3-0.6B"
BATCH_SIZES="1"
NUM_SPEC_TOKENS="7"
GPU_MEM_UTIL="0.90"
MAX_MODEL_LEN="8192"

# TARGET_CFG: 4b (3h, 1 GPU), 8b (6h, 1 GPU), 30b (10h, 2 GPU)
case "${TARGET_CFG:-4b}" in
  4b)
    TARGET_MODELS="Qwen/Qwen3-4B"
    TP_MAP="Qwen/Qwen3-4B=1"
    RESULTS_CSV="spec_decode_metrics_4b.csv"
    RESULTS_JSONL="spec_decode_metrics_4b.jsonl"
    ;;
  8b)
    TARGET_MODELS="Qwen/Qwen3-8B"
    TP_MAP="Qwen/Qwen3-8B=1"
    RESULTS_CSV="spec_decode_metrics_8b.csv"
    RESULTS_JSONL="spec_decode_metrics_8b.jsonl"
    ;;
  30b)
    TARGET_MODELS="Qwen/Qwen3-30B-A3B"
    TP_MAP="Qwen/Qwen3-30B-A3B=2"
    RESULTS_CSV="spec_decode_metrics_30b.csv"
    RESULTS_JSONL="spec_decode_metrics_30b.jsonl"
    ;;
  *)
    echo "Unknown TARGET_CFG=${TARGET_CFG}. Use 4b, 8b, or 30b."
    exit 1
    ;;
esac

# Hugging Face / cache
CACHE_ROOT="${SCRATCH:-/home/junsuk87/scratch}/vllm_job_cache/${SLURM_JOB_ID}"
mkdir -p "${CACHE_ROOT}"/{hf_home,hf_datasets,hf_hub,xdg,vllm,torchinductor,triton,tmp,uv}
export HF_HOME="${CACHE_ROOT}/hf_home"
export HF_DATASETS_CACHE="${CACHE_ROOT}/hf_datasets"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/hf_hub"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torchinductor"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export TMPDIR="${CACHE_ROOT}/tmp"
export UV_CACHE_DIR="${CACHE_ROOT}/uv"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# -----------------------------
# Run
# -----------------------------
mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"
source "${VENV_DIR}/bin/activate"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1

echo "================ JOB INFO ================"
echo "JOB_ID: ${SLURM_JOB_ID}"
echo "TARGET_CFG: ${TARGET_CFG:-4b}"
echo "HOSTNAME: $(hostname)"
echo "DATE: $(date)"
echo "PWD: $(pwd)"
echo "PYTHON: $(which python)"
echo "HF_HOME: ${HF_HOME}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "=========================================="

python -u scripts/run_spec_decode_metrics.py \
  --profile-time \
  --draft-model "${DRAFT_MODEL}" \
  --target-models "${TARGET_MODELS}" \
  --tp-map "${TP_MAP}" \
  --trust-remote-code \
  --enable-chunked-prefill \
  --datasets aime25,codeelo \
  --batch-sizes "${BATCH_SIZES}" \
  --num-spec-tokens "${NUM_SPEC_TOKENS}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --results-csv "${RESULTS_CSV}" \
  --results-jsonl "${RESULTS_JSONL}"

echo "Done. Results: ${RESULTS_CSV} ${RESULTS_JSONL}"
