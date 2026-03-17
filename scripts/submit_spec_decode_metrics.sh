#!/bin/bash
# Usage:
#   bash scripts/submit_spec_decode_metrics.sh
#
# Optional:
#   MANIFEST=scripts/spec_decode_pairs_manifest.tsv bash scripts/submit_spec_decode_metrics.sh
#   PAIR_FILTER='llama32_' bash scripts/submit_spec_decode_metrics.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
SLURM_SCRIPT="${SLURM_SCRIPT:-${REPO_DIR}/scripts/slurm_run_spec_decode_metrics.slurm}"
MANIFEST="${MANIFEST:-${REPO_DIR}/scripts/spec_decode_pairs_manifest.tsv}"
PAIR_FILTER="${PAIR_FILTER:-}"

mkdir -p "${REPO_DIR}/logs"

if [[ ! -f "${SLURM_SCRIPT}" ]]; then
  echo "Slurm script not found: ${SLURM_SCRIPT}"
  exit 1
fi

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Manifest not found: ${MANIFEST}"
  exit 1
fi

echo "Submitting speculative decoding jobs from ${MANIFEST}"
echo "Results root: ${REPO_DIR}/results/profile"
if [[ -n "${PAIR_FILTER}" ]]; then
  echo "PAIR_FILTER: ${PAIR_FILTER}"
fi
echo

submitted=0
while IFS=$'\t' read -r pair_id draft_model target_model tp_size gpu_count time_limit note; do
  [[ -z "${pair_id}" ]] && continue
  [[ "${pair_id}" =~ ^# ]] && continue
  if [[ -n "${PAIR_FILTER}" && "${pair_id}" != *"${PAIR_FILTER}"* ]]; then
    continue
  fi

  job_name="spec_${pair_id}"
  export_str="ALL,PAIR_ID=${pair_id},DRAFT_MODEL=${draft_model},TARGET_MODEL=${target_model},TP_SIZE=${tp_size},RESULTS_ROOT=${REPO_DIR}/results/profile"

  job_id=$(sbatch \
    --job-name="${job_name}" \
    --time="${time_limit}" \
    --gres="gpu:h100:${gpu_count}" \
    --export="${export_str}" \
    "${SLURM_SCRIPT}" \
    | awk '{print $4}')

  printf 'Submitted %-32s job=%s  gpus=%s  time=%s\n' "${pair_id}" "${job_id}" "${gpu_count}" "${time_limit}"
  ((submitted+=1))
done < "${MANIFEST}"

echo
echo "Total submitted jobs: ${submitted}"
echo "Logs: ${REPO_DIR}/logs/spec_decode_metrics_*_<jobid>.out"
echo "Results: ${REPO_DIR}/results/profile"
