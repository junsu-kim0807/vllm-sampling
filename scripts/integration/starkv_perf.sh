#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Simple integration harness to compare vanilla vLLM vs StarKV-enabled runs
# across a few scenarios (single batch, batch scaling, reforward stress,
# confidence threshold sweep).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MODEL=${MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}
COMMON_ARGS=(
  --backend vllm
  --model "$MODEL"
  --load-format dummy
  --num-prompts 64
  --input-len 1024
  --output-len 32
  --enforce-eager
  --output-json
)
RESULTS_FILE="${RESULTS_FILE:-starkv_perf_results.csv}"
mkdir -p scripts/integration/results

echo "scenario,backend,requests_per_second,tokens_per_second,output_tokens_per_second,elapsed_time" > "$RESULTS_FILE"

run_case() {
  local scenario=$1
  shift
  local json_file="scripts/integration/results/${scenario}.json"
  python -m vllm.entrypoints.cli.main bench throughput \
    "${COMMON_ARGS[@]}" "$json_file" \
    "$@"

  python - <<'PY' "$RESULTS_FILE" "$json_file" "$scenario"
import json, sys
results_file, json_file, scenario = sys.argv[1:]
with open(json_file) as f:
    data = json.load(f)
row = [
    scenario,
    data.get("backend", "vllm"),
    f'{data["requests_per_second"]:.4f}',
    f'{data["tokens_per_second"]:.4f}',
    f'{data.get("output_tokens_per_second", 0.0):.4f}',
    f'{data["elapsed_time"]:.4f}',
]
with open(results_file, "a") as out:
    out.write(",".join(row) + "\n")
PY
}

# 1) Single-batch baseline comparisons
run_case "baseline_vllm" \
  --num-prompts 1 --input-len 4096 --output-len 64

run_case "starkv_single" \
  --num-prompts 1 --input-len 4096 --output-len 64 \
  --enable-starkv-super-cache \
  --starkv-score-fn morphkv \
  --starkv-compression-ratio 0.5 \
  --starkv-confidence-threshold 0.8

# 2) Batch-size scaling
for sizes in "1" "1 2" "1 2 4 8"; do
  run_case "starkv_scaling_${sizes// /_}" \
    --enable-starkv-super-cache \
    --starkv-score-fn morphkv \
    --starkv-compression-ratio 0.5 \
    --starkv-batch-sizes $sizes
done

# 3) Reforward stress (toggle STARkV_FORCE_LOW_CONF)
STARKV_FORCE_LOW_CONF=0 run_case "starkv_reforward_low" \
  --enable-starkv-super-cache \
  --starkv-score-fn morphkv \
  --starkv-compression-ratio 0.5 \
  --starkv-confidence-threshold 0.2

STARKV_FORCE_LOW_CONF=1 run_case "starkv_reforward_high" \
  --enable-starkv-super-cache \
  --starkv-score-fn morphkv \
  --starkv-compression-ratio 0.5 \
  --starkv-confidence-threshold 0.8

# 4) Confidence sweep
for threshold in 0.2 0.5 0.8 0.95; do
  run_case "starkv_conf_${threshold}" \
    --enable-starkv-super-cache \
    --starkv-score-fn morphkv \
    --starkv-compression-ratio 0.5 \
    --starkv-confidence-threshold "$threshold"
done

echo "Results written to $RESULTS_FILE"


