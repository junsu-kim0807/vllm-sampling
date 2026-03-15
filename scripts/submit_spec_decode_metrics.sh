#!/bin/bash
# Submit three spec_decode_metrics jobs in parallel:
#   - 0.6B+4B:  3h, 1 GPU  -> spec_decode_metrics_4b.csv / .jsonl
#   - 0.6B+8B:  6h, 1 GPU  -> spec_decode_metrics_8b.csv / .jsonl
#   - 0.6B+30B: 10h, 2 GPU -> spec_decode_metrics_30b.csv / .jsonl
# Usage: from repo root, run: bash scripts/submit_spec_decode_metrics.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
SLURM_SCRIPT="${REPO_DIR}/scripts/slurm_run_spec_decode_metrics.slurm"
mkdir -p "${REPO_DIR}/logs"

if [[ ! -f "${SLURM_SCRIPT}" ]]; then
  echo "Slurm script not found: ${SLURM_SCRIPT}"
  exit 1
fi

echo "Submitting 3 jobs (0.6B+4B, 0.6B+8B, 0.6B+30B) from ${REPO_DIR}"

JOB_4B=$(sbatch \
  --job-name=spec_metrics_4b \
  --time=3:00:00 \
  --gres=gpu:h100:1 \
  --export=ALL,TARGET_CFG=4b \
  "${SLURM_SCRIPT}" \
  | awk '{print $4}')
echo "  Submitted 0.6B+4B  (3h, 1 GPU): job ${JOB_4B}"

JOB_8B=$(sbatch \
  --job-name=spec_metrics_8b \
  --time=6:00:00 \
  --gres=gpu:h100:1 \
  --export=ALL,TARGET_CFG=8b \
  "${SLURM_SCRIPT}" \
  | awk '{print $4}')
echo "  Submitted 0.6B+8B  (6h, 1 GPU): job ${JOB_8B}"

JOB_30B=$(sbatch \
  --job-name=spec_metrics_30b \
  --time=10:00:00 \
  --gres=gpu:h100:2 \
  --export=ALL,TARGET_CFG=30b \
  "${SLURM_SCRIPT}" \
  | awk '{print $4}')
echo "  Submitted 0.6B+30B (10h, 2 GPU): job ${JOB_30B}"

echo ""
echo "Results will be written under ${REPO_DIR}:"
echo "  spec_decode_metrics_4b.csv / spec_decode_metrics_4b.jsonl"
echo "  spec_decode_metrics_8b.csv / spec_decode_metrics_8b.jsonl"
echo "  spec_decode_metrics_30b.csv / spec_decode_metrics_30b.jsonl"
echo "Logs: ${REPO_DIR}/logs/spec_decode_metrics_*_<jobid>.out"
