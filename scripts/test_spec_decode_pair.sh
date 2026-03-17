#!/bin/bash
# Interactive sanity test on a tiny subset.
#
# Examples:
#   bash scripts/test_spec_decode_pair.sh
#   DRAFT_MODEL="Qwen/Qwen3-0.6B" TARGET_MODEL="Qwen/Qwen3-8B" TP_SIZE=1 bash scripts/test_spec_decode_pair.sh
#   DRAFT_MODEL="meta-llama/Llama-3.2-1B-Instruct" TARGET_MODEL="meta-llama/Llama-3.2-3B-Instruct" TP_SIZE=1 DATASETS="aime25" bash scripts/test_spec_decode_pair.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

export DRAFT_MODEL="${DRAFT_MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
export TARGET_MODEL="${TARGET_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
export TP_SIZE="${TP_SIZE:-1}"
export DATASETS="${DATASETS:-aime25}"
export BATCH_SIZES="${BATCH_SIZES:-1}"
export NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-7}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
export PROFILE_TIME="${PROFILE_TIME:-1}"
export WARMUP_ITERS="${WARMUP_ITERS:-1}"
export WARMUP_MAX_TOKENS="${WARMUP_MAX_TOKENS:-16}"
export MAX_SAMPLES_AIME25="${MAX_SAMPLES_AIME25:-4}"
export MAX_SAMPLES_CODEELO="${MAX_SAMPLES_CODEELO:-2}"
export AIME_MAX_NEW_TOKENS="${AIME_MAX_NEW_TOKENS:-64}"
export CODEELO_MAX_NEW_TOKENS="${CODEELO_MAX_NEW_TOKENS:-256}"
export RESULTS_ROOT="${RESULTS_ROOT:-${REPO_DIR}/results/profile_test}"
export VERBOSE="${VERBOSE:-1}"

cd "${REPO_DIR}"
bash scripts/run_spec_decode_pair.sh
