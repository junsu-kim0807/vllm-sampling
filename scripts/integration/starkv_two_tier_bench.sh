#!/usr/bin/env bash
set -euo pipefail

# StarKV two-tier (super/sub) benchmark runner
#
# - Mode A: super tier on GPU (default) -> do NOT pass --starkv-offload
# - Mode B: super tier on CPU pinned -> pass --starkv-offload
#
# This script runs throughput benchmark(s) and captures:
# - JSON outputs
# - full logs
# - reforward trigger counts (from logs)
#
# Usage:
#   cd vllm-sampling
#   bash scripts/integration/starkv_two_tier_bench.sh
#
# Optional env overrides:
#   MODEL=... NUM_PROMPTS=... INPUT_LEN=... OUTPUT_LEN=...
#   THRESH=0.99
#   PYTHON=python
#   EXTRA_ARGS="--gpu-memory-utilization 0.9"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
NUM_PROMPTS="${NUM_PROMPTS:-32}"
INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-64}"
THRESH="${THRESH:-0.99}"

EXTRA_ARGS="${EXTRA_ARGS:-}"

STAMP="$(date +"%Y%m%d_%H%M%S")"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/bench_outputs/starkv_two_tier_${STAMP}}"
mkdir -p "$OUT_DIR"

COMMON_ARGS=(
  --backend vllm
  --model "$MODEL"
  --load-format dummy
  --dataset-name random
  --num-prompts "$NUM_PROMPTS"
  --input-len "$INPUT_LEN"
  --output-len "$OUTPUT_LEN"
  --enforce-eager
)

run_case () {
  local name="$1"; shift
  local json_out="$OUT_DIR/${name}.json"
  local log_out="$OUT_DIR/${name}.log"

  echo "==> Running: ${name}"
  echo "    json: $json_out"
  echo "    log : $log_out"

  # Show reforward triggers at INFO-level.
  STARKV_LOG_REFORWARD=1 \
  "$PYTHON" -m vllm.entrypoints.cli.main bench throughput \
    "${COMMON_ARGS[@]}" \
    --output-json "$json_out" \
    "$@" \
    ${EXTRA_ARGS} \
    2>&1 | tee "$log_out"

  local cnt
  cnt="$(grep -c "StarKV decode-confidence reforward trigger" "$log_out" || true)"
  echo "==> ${name}: reforward-trigger-log-count = ${cnt}"
  echo "${cnt}" > "$OUT_DIR/${name}.reforward_count.txt"
}

echo "Output directory: $OUT_DIR"
echo "Model: $MODEL"
echo "Prompts: $NUM_PROMPTS, input_len: $INPUT_LEN, output_len: $OUTPUT_LEN, threshold: $THRESH"
echo

# 0) Vanilla (baseline)
run_case "vanilla" \
  --starkv-confidence-threshold "$THRESH"

# 1) StarKV enabled, two-tier super on GPU (no offload flag)
run_case "starkv_super_gpu" \
  --enable-starkv-super-cache \
  --starkv-confidence-threshold "$THRESH"

# 2) StarKV enabled, two-tier super on CPU pinned (offload flag)
run_case "starkv_super_cpu" \
  --enable-starkv-super-cache \
  --starkv-confidence-threshold "$THRESH" \
  --starkv-offload

echo
echo "Done."
echo "Artifacts:"
ls -1 "$OUT_DIR"


