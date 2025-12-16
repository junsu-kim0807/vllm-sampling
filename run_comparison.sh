#!/usr/bin/env bash
# StarKV vs Vanilla vLLM 자동 비교 스크립트

set -euo pipefail

# 설정
MODEL=${MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}
RESULTS_DIR="benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${RESULTS_DIR}/comparison_${TIMESTAMP}.csv"
LOG_FILE="${RESULTS_DIR}/benchmark_${TIMESTAMP}.log"

# 결과 디렉토리 생성
mkdir -p "$RESULTS_DIR"

# 로그 함수
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "Starting benchmark comparison: Vanilla vLLM vs StarKV"
log "Model: $MODEL"
log "Results will be saved to: $RESULTS_FILE"

# CSV 헤더 작성
echo "scenario,variant,requests_per_second,tokens_per_second,output_tokens_per_second,elapsed_time,starkv_reforward_requests,starkv_reforward_tokens" > "$RESULTS_FILE"

# 공통 인자
COMMON_ARGS=(
  --backend vllm
  --model "$MODEL"
  --load-format dummy
  --dataset-name random
  --enforce-eager
  --output-json
)

# StarKV 인자
STARKV_ARGS=(
  --enable-starkv-super-cache
  --starkv-score-fn "${STARKV_SCORE_FN:-morphkv}"
  --starkv-compression-ratio "${STARKV_COMPRESSION_RATIO:-0.5}"
  --starkv-confidence-threshold "${STARKV_CONFIDENCE_THRESHOLD:-0.8}"
  --starkv-offload
)

# 결과 추가 함수
append_results() {
  local scenario=$1
  local variant=$2
  local json_file=$3

  python3 <<'PYTHON_SCRIPT' "$RESULTS_FILE" "$json_file" "$scenario" "$variant"
import json
import sys

results_file, json_path, scenario, variant = sys.argv[1:]

try:
    with open(json_path) as f:
        data = json.load(f)
    
    stats = data.get("starkv_stats") or {}
    row = [
        scenario,
        variant,
        f'{data.get("requests_per_second", 0.0):.4f}',
        f'{data.get("tokens_per_second", 0.0):.4f}',
        f'{data.get("output_tokens_per_second", 0.0):.4f}',
        f'{data.get("elapsed_time", 0.0):.4f}',
        str(stats.get("total_reforward_requests", 0)),
        str(stats.get("total_reforward_tokens", 0)),
    ]
    
    with open(results_file, "a") as out:
        out.write(",".join(row) + "\n")
except Exception as e:
    print(f"Error processing {json_path}: {e}", file=sys.stderr)
PYTHON_SCRIPT
}

# 벤치마크 실행 함수
run_variant() {
  local scenario=$1
  local variant=$2
  shift 2
  local json_path="${RESULTS_DIR}/${scenario}_${variant}_${TIMESTAMP}.json"
  
  log "Running: $scenario - $variant"
  
  python -m vllm.entrypoints.cli.main bench throughput \
    "${COMMON_ARGS[@]}" \
    --output-json "$json_path" \
    "$@" 2>&1 | tee -a "$LOG_FILE"
  
  if [ -f "$json_path" ]; then
    append_results "$scenario" "$variant" "$json_path"
    log "Completed: $scenario - $variant"
  else
    log "ERROR: Results file not found: $json_path"
  fi
}

# 시나리오 비교 함수
compare_scenario() {
  local scenario=$1
  shift
  log ""
  log "=== Scenario: $scenario ==="
  
  # Vanilla vLLM 실행
  run_variant "$scenario" "vllm" "$@"
  
  # StarKV 실행
  run_variant "$scenario" "starkv" "$@" "${STARKV_ARGS[@]}"
}

# 시나리오 1: Single long-prefill (latency stress)
compare_scenario "single_prefill" \
  --num-prompts 1 \
  --input-len 4096 \
  --output-len 64

# 시나리오 2: Medium batch (steady-state throughput)
compare_scenario "batch_32" \
  --num-prompts 32 \
  --input-len 1024 \
  --output-len 64

# 시나리오 3: Large batch (scaling test)
compare_scenario "batch_128" \
  --num-prompts 128 \
  --input-len 512 \
  --output-len 32

log ""
log "Benchmark comparison complete!"
log "Results saved to: $RESULTS_FILE"
log "Log file: $LOG_FILE"

