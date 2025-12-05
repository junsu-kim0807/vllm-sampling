#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Quick harness to compare plain vLLM vs StarKV-enabled runs under
# identical workloads. Results land in CSV so you can diff RPS and
# reforward metrics at a glance.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MODEL=${MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}
RESULTS_DIR="scripts/integration/results"
mkdir -p "$RESULTS_DIR"
RESULTS_FILE="${RESULTS_DIR}/vllm_vs_starkv.csv"

COMMON_ARGS=(
  --backend vllm
  --model "$MODEL"
  --load-format dummy
  --dataset-name random
  --num-prompts 64
  --input-len 1024
  --output-len 32
  --enforce-eager
  --output-json
)

STARKV_ARGS=(
  --enable-starkv-super-cache
  --starkv-score-fn "${STARKV_SCORE_FN:-morphkv}"
  --starkv-compression-ratio "${STARKV_COMPRESSION_RATIO:-0.5}"
  --starkv-confidence-threshold "${STARKV_CONFIDENCE_THRESHOLD:-0.8}"
  --starkv-offload
)

echo "scenario,variant,requests_per_second,tokens_per_second,output_tokens_per_second,elapsed_time,starkv_reforward_requests,starkv_reforward_tokens" > "$RESULTS_FILE"

append_results() {
  local scenario=$1
  local variant=$2
  local json_file=$3

  python - <<'PY' "$RESULTS_FILE" "$json_file" "$scenario" "$variant"
import json, sys
results_file, json_path, scenario, variant = sys.argv[1:]
with open(json_path) as f:
    data = json.load(f)
stats = data.get("starkv_stats") or {}
row = [
    scenario,
    variant,
    f'{data["requests_per_second"]:.4f}',
    f'{data["tokens_per_second"]:.4f}',
    f'{data.get("output_tokens_per_second", 0.0):.4f}',
    f'{data["elapsed_time"]:.4f}',
    str(stats.get("total_reforward_requests", 0)),
    str(stats.get("total_reforward_tokens", 0)),
]
with open(results_file, "a") as out:
    out.write(",".join(row) + "\n")
PY
}

run_variant() {
  local scenario=$1
  local variant=$2
  shift 2
  local json_path="${RESULTS_DIR}/${scenario}_${variant}.json"
  python -m vllm.entrypoints.cli.main bench throughput \
    "${COMMON_ARGS[@]}" "$json_path" \
    "$@"
  append_results "$scenario" "$variant" "$json_path"
}

compare_scenario() {
  local scenario=$1
  shift
  printf "\n=== Scenario: %s ===\n" "$scenario"
  run_variant "$scenario" "vllm" "$@"
  run_variant "$scenario" "starkv" "$@" "${STARKV_ARGS[@]}"
}

# 1) Single long-prefill run (stress latency)
compare_scenario "single_prefill" \
  --num-prompts 1 \
  --input-len 4096 \
  --output-len 64

# 2) Medium batch steady-state throughput
compare_scenario "batch_32" \
  --num-prompts 32 \
  --input-len 1024 \
  --output-len 64

# 3) Large batch scaling
compare_scenario "batch_128" \
  --num-prompts 128 \
  --input-len 512 \
  --output-len 32

echo
echo "Comparison complete -> ${RESULTS_FILE}"
echo "Columns include StarKV reforward stats so you can gauge savings per load."

