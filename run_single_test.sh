#!/usr/bin/env bash
# 단일 테스트 실행 스크립트

set -euo pipefail

SCENARIO=${1:-batch_32}
VARIANT=${2:-vllm}  # vllm or starkv
MODEL=${MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}
RESULTS_DIR="benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$RESULTS_DIR"

# 시나리오별 설정
case "$SCENARIO" in
  single_prefill)
    ARGS=(--num-prompts 1 --input-len 4096 --output-len 64)
    ;;
  batch_32)
    ARGS=(--num-prompts 32 --input-len 1024 --output-len 64)
    ;;
  batch_128)
    ARGS=(--num-prompts 128 --input-len 512 --output-len 32)
    ;;
  *)
    echo "Unknown scenario: $SCENARIO"
    echo "Available scenarios: single_prefill, batch_32, batch_128"
    exit 1
    ;;
esac

# StarKV 인자 추가
if [ "$VARIANT" = "starkv" ]; then
  ARGS+=(
    --enable-starkv-super-cache
    --starkv-score-fn "${STARKV_SCORE_FN:-morphkv}"
    --starkv-compression-ratio "${STARKV_COMPRESSION_RATIO:-0.5}"
    --starkv-confidence-threshold "${STARKV_CONFIDENCE_THRESHOLD:-0.8}"
    --starkv-offload
  )
fi

JSON_OUTPUT="${RESULTS_DIR}/${SCENARIO}_${VARIANT}_${TIMESTAMP}.json"

echo "Running: $SCENARIO - $VARIANT"
echo "Output: $JSON_OUTPUT"

python -m vllm.entrypoints.cli.main bench throughput \
  --backend vllm \
  --model "$MODEL" \
  --load-format dummy \
  --dataset-name random \
  --enforce-eager \
  --output-json "$JSON_OUTPUT" \
  "${ARGS[@]}"

echo "Results saved to: $JSON_OUTPUT"

