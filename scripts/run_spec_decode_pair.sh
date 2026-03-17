#!/bin/bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/project/6045356/junsu/kv_cache/vllm-sampling}"
VENV_DIR="${VENV_DIR:-/project/6045356/junsu/kv_cache/.venv}"

DRAFT_MODEL="${DRAFT_MODEL:?DRAFT_MODEL is required}"
TARGET_MODEL="${TARGET_MODEL:?TARGET_MODEL is required}"
TP_SIZE="${TP_SIZE:?TP_SIZE is required}"

BATCH_SIZES="${BATCH_SIZES:-1}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-7}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
DATASETS="${DATASETS:-aime25,codeelo}"
DTYPE="${DTYPE:-auto}"
SEED="${SEED:-0}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_DIR}/results/profile}"
PROFILE_TIME="${PROFILE_TIME:-1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-1}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
DISABLE_LOG_STATS="${DISABLE_LOG_STATS:-0}"
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-0}"
WARMUP_ITERS="${WARMUP_ITERS:-2}"
WARMUP_MAX_TOKENS="${WARMUP_MAX_TOKENS:-32}"
MAX_SAMPLES_AIME25="${MAX_SAMPLES_AIME25:-}"
MAX_SAMPLES_CODEELO="${MAX_SAMPLES_CODEELO:-}"
AIME_MAX_NEW_TOKENS="${AIME_MAX_NEW_TOKENS:-256}"
CODEELO_MAX_NEW_TOKENS="${CODEELO_MAX_NEW_TOKENS:-1024}"
VERBOSE="${VERBOSE:-0}"

mkdir -p "${REPO_DIR}/logs" "${RESULTS_ROOT}"
cd "${REPO_DIR}"
source "${VENV_DIR}/bin/activate"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

pair_slug="$(python - <<'PY'
import os, re
draft = os.environ["DRAFT_MODEL"]
target = os.environ["TARGET_MODEL"]
def sanitize(s: str) -> str:
    s = s.strip().replace("/", "__")
    s = re.sub(r"[^A-Za-z0-9._+@=,]+", "_", s)
    return s.strip("._")
print(f"{sanitize(draft)}__TO__{sanitize(target)}")
PY
)"

RESULTS_CSV="${RESULTS_CSV:-aggregate__${pair_slug}.csv}"
RESULTS_JSONL="${RESULTS_JSONL:-aggregate__${pair_slug}.jsonl}"

cmd=(
  python -u scripts/run_spec_decode_metrics.py
  --draft-model "${DRAFT_MODEL}"
  --target-models "${TARGET_MODEL}"
  --tp-map "${TARGET_MODEL}=${TP_SIZE}"
  --datasets "${DATASETS}"
  --batch-sizes "${BATCH_SIZES}"
  --num-spec-tokens "${NUM_SPEC_TOKENS}"
  --gpu-memory-utilization "${GPU_MEM_UTIL}"
  --max-model-len "${MAX_MODEL_LEN}"
  --dtype "${DTYPE}"
  --seed "${SEED}"
  --warmup-iters "${WARMUP_ITERS}"
  --warmup-max-tokens "${WARMUP_MAX_TOKENS}"
  --aime-max-new-tokens "${AIME_MAX_NEW_TOKENS}"
  --codeelo-max-new-tokens "${CODEELO_MAX_NEW_TOKENS}"
  --results-root "${RESULTS_ROOT}"
  --results-csv "${RESULTS_CSV}"
  --results-jsonl "${RESULTS_JSONL}"
)

if [[ "${PROFILE_TIME}" == "1" ]]; then
  cmd+=(--profile-time)
fi
if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  cmd+=(--trust-remote-code)
fi
if [[ "${ENABLE_CHUNKED_PREFILL}" == "1" ]]; then
  cmd+=(--enable-chunked-prefill)
fi
if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  cmd+=(--enforce-eager)
fi
if [[ "${DISABLE_LOG_STATS}" == "1" ]]; then
  cmd+=(--disable-log-stats)
fi
if [[ "${DISABLE_CUSTOM_ALL_REDUCE}" == "1" ]]; then
  cmd+=(--disable-custom-all-reduce)
fi
if [[ -n "${MAX_SAMPLES_AIME25}" ]]; then
  cmd+=(--max-samples-aime25 "${MAX_SAMPLES_AIME25}")
fi
if [[ -n "${MAX_SAMPLES_CODEELO}" ]]; then
  cmd+=(--max-samples-codeelo "${MAX_SAMPLES_CODEELO}")
fi
if [[ "${VERBOSE}" == "1" ]]; then
  cmd+=(--verbose)
fi

echo "============================================================"
echo "PAIR: ${pair_slug}"
echo "DRAFT_MODEL: ${DRAFT_MODEL}"
echo "TARGET_MODEL: ${TARGET_MODEL}"
echo "TP_SIZE: ${TP_SIZE}"
echo "RESULTS_ROOT: ${RESULTS_ROOT}"
echo "RESULTS_CSV: ${RESULTS_CSV}"
echo "RESULTS_JSONL: ${RESULTS_JSONL}"
echo "PWD: $(pwd)"
echo "PYTHON: $(which python)"
echo "============================================================"

printf '%q ' "${cmd[@]}"
echo
"${cmd[@]}"
