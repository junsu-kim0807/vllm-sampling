#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOBS_ROOT="${REPO_ROOT}/scripts/jobs/spec_decode"

PAIR_FILTER="${PAIR_FILTER:-}"
DATASET_FILTER="${DATASET_FILTER:-}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/submit_spec_decode_jobs.sh
  PAIR_FILTER="llama32_1b" bash scripts/submit_spec_decode_jobs.sh
  DATASET_FILTER="aime25" bash scripts/submit_spec_decode_jobs.sh
  DRY_RUN=1 bash scripts/submit_spec_decode_jobs.sh

Environment variables:
  PAIR_FILTER     Submit only jobs whose pair_id contains this string
  DATASET_FILTER  Submit only jobs whose dataset name contains this string
  DRY_RUN         If set to 1, only print matching scripts without submitting
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -d "${JOBS_ROOT}" ]]; then
  echo "Jobs directory not found: ${JOBS_ROOT}"
  exit 1
fi

mapfile -t job_scripts < <(find "${JOBS_ROOT}" -type f -name "*.slurm" | sort)

if [[ ${#job_scripts[@]} -eq 0 ]]; then
  echo "No .slurm files found under ${JOBS_ROOT}"
  exit 1
fi

matched=()
for script in "${job_scripts[@]}"; do
  pair_id="$(basename "$(dirname "${script}")")"
  dataset="$(basename "${script}" .slurm)"

  if [[ -n "${PAIR_FILTER}" && "${pair_id}" != *"${PAIR_FILTER}"* ]]; then
    continue
  fi

  if [[ -n "${DATASET_FILTER}" && "${dataset}" != *"${DATASET_FILTER}"* ]]; then
    continue
  fi

  matched+=("${script}")
done

if [[ ${#matched[@]} -eq 0 ]]; then
  echo "No matching jobs found."
  echo "PAIR_FILTER=${PAIR_FILTER:-<empty>}"
  echo "DATASET_FILTER=${DATASET_FILTER:-<empty>}"
  exit 1
fi

echo "Found ${#matched[@]} matching job scripts."
echo

submitted=0
for script in "${matched[@]}"; do
  pair_id="$(basename "$(dirname "${script}")")"
  dataset="$(basename "${script}" .slurm)"

  job_name="$(grep -E '^#SBATCH --job-name=' "${script}" | head -n1 | cut -d= -f2- || true)"
  time_limit="$(grep -E '^#SBATCH --time=' "${script}" | head -n1 | cut -d= -f2- || true)"
  gres="$(grep -E '^#SBATCH --gres=' "${script}" | head -n1 | cut -d= -f2- || true)"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY_RUN] pair=${pair_id} dataset=${dataset} job_name=${job_name} time=${time_limit} gres=${gres}"
    echo "          ${script}"
    echo
    continue
  fi

  submit_out="$(sbatch "${script}")"
  job_id="$(awk '{print $4}' <<< "${submit_out}")"

  echo "Submitted pair=${pair_id} dataset=${dataset} job=${job_id} time=${time_limit} gres=${gres}"
  echo "          ${script}"
  submitted=$((submitted + 1))
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo
  echo "Dry run complete. ${#matched[@]} scripts matched."
else
  echo
  echo "Done. Submitted ${submitted} jobs."
fi