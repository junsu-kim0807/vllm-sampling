#!/usr/bin/env bash
set -euo pipefail

# LongBench-v1 evaluation runner (vLLM + optional StarKV)
#
# This script runs generation for each LongBench-v1 task and (optionally) computes metrics
# using kvpress-baseline's LongBench scorer implementation.
#
# Requirements (typical):
#   pip install datasets pandas rouge jieba fuzzywuzzy python-Levenshtein
#
# Usage:
#   cd vllm-sampling
#   bash scripts/longbench/run_longbench_v1.sh
#
# Env overrides:
#   MODEL=... OUT_DIR=... TASKS=all|comma,separated
#   THRESH=0.99 OFFLOAD=0|1 ENABLE_STARKV=0|1
#   BATCH_SIZE=4 MAX_MODEL_LEN=131072 GPU_MEM_UTIL=0.9

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
TASKS="${TASKS:-all}"
REPO="${REPO:-Xnhyacinth/LongBench}"
SPLIT="${SPLIT:-test}"

ENABLE_STARKV="${ENABLE_STARKV:-1}"
THRESH="${THRESH:-0.99}"
OFFLOAD="${OFFLOAD:-0}"

BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"

OUT_DIR="${OUT_DIR:-$ROOT_DIR/longbench_outputs/$(date +"%Y%m%d_%H%M%S")}"

ARGS=(
  --model "$MODEL"
  --repo "$REPO"
  --split "$SPLIT"
  --tasks "$TASKS"
  --out-dir "$OUT_DIR"
  --batch-size "$BATCH_SIZE"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
)

if [[ "$ENFORCE_EAGER" == "1" ]]; then
  ARGS+=(--enforce-eager)
fi

if [[ "$ENABLE_STARKV" == "1" ]]; then
  ARGS+=(--enable-starkv-super-cache --starkv-confidence-threshold "$THRESH")
  if [[ "$OFFLOAD" == "1" ]]; then
    ARGS+=(--starkv-offload)
  fi
fi

echo "OUT_DIR=$OUT_DIR"
echo "MODEL=$MODEL"
echo "TASKS=$TASKS"
echo "ENABLE_STARKV=$ENABLE_STARKV THRESH=$THRESH OFFLOAD=$OFFLOAD"

# Optional: print reforward triggers
export STARKV_LOG_REFORWARD="${STARKV_LOG_REFORWARD:-0}"

"$PYTHON" scripts/longbench/longbench_v1_eval_vllm.py "${ARGS[@]}"


