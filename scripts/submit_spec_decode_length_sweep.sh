#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/Users/junsu/EECE571/vllm-nips26"

echo "sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/speculative/b256/k5/llama32_1b_to_llama33_70b/gov_report.slurm"
sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/speculative/b256/k5/llama32_1b_to_llama33_70b/gov_report.slurm
echo "sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/speculative/b256/k5/vicuna_68m_to_vicuna_13b_v13/gov_report.slurm"
sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/speculative/b256/k5/vicuna_68m_to_vicuna_13b_v13/gov_report.slurm
echo "sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/magicdec/b256/k5/llama32_1b_to_llama33_70b/gov_report.slurm"
sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/magicdec/b256/k5/llama32_1b_to_llama33_70b/gov_report.slurm
echo "sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/magicdec/b256/k5/vicuna_68m_to_vicuna_13b_v13/gov_report.slurm"
sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/magicdec/b256/k5/vicuna_68m_to_vicuna_13b_v13/gov_report.slurm
echo "sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/tetris/b256/k5/llama32_1b_to_llama33_70b/gov_report.slurm"
sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/tetris/b256/k5/llama32_1b_to_llama33_70b/gov_report.slurm
echo "sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/tetris/b256/k5/vicuna_68m_to_vicuna_13b_v13/gov_report.slurm"
sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/tetris/b256/k5/vicuna_68m_to_vicuna_13b_v13/gov_report.slurm
echo "sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/pivot/pivot/b256/k5/llama32_1b_to_llama33_70b/gov_report.slurm"
sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/pivot/pivot/b256/k5/llama32_1b_to_llama33_70b/gov_report.slurm
echo "sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/pivot/pivot/b256/k5/vicuna_68m_to_vicuna_13b_v13/gov_report.slurm"
sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/pivot/pivot/b256/k5/vicuna_68m_to_vicuna_13b_v13/gov_report.slurm
echo "sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/pivot/pivot_spechive/b256/k5/llama32_1b_to_llama33_70b/gov_report.slurm"
sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/pivot/pivot_spechive/b256/k5/llama32_1b_to_llama33_70b/gov_report.slurm
echo "sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/pivot/pivot_spechive/b256/k5/vicuna_68m_to_vicuna_13b_v13/gov_report.slurm"
sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/pivot/pivot_spechive/b256/k5/vicuna_68m_to_vicuna_13b_v13/gov_report.slurm
echo "sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/speculative/b256/k7/llama32_1b_to_llama33_70b/gov_report.slurm"
sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/speculative/b256/k7/llama32_1b_to_llama33_70b/gov_report.slurm
echo "sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/speculative/b256/k7/vicuna_68m_to_vicuna_13b_v13/gov_report.slurm"
sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/speculative/b256/k7/vicuna_68m_to_vicuna_13b_v13/gov_report.slurm
echo "sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/magicdec/b256/k7/llama32_1b_to_llama33_70b/gov_report.slurm"
sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/magicdec/b256/k7/llama32_1b_to_llama33_70b/gov_report.slurm
echo "sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/magicdec/b256/k7/vicuna_68m_to_vicuna_13b_v13/gov_report.slurm"
sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/magicdec/b256/k7/vicuna_68m_to_vicuna_13b_v13/gov_report.slurm
echo "sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/tetris/b256/k7/llama32_1b_to_llama33_70b/gov_report.slurm"
sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/tetris/b256/k7/llama32_1b_to_llama33_70b/gov_report.slurm
echo "sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/tetris/b256/k7/vicuna_68m_to_vicuna_13b_v13/gov_report.slurm"
sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/tetris/b256/k7/vicuna_68m_to_vicuna_13b_v13/gov_report.slurm
echo "sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/pivot/pivot/b256/k7/llama32_1b_to_llama33_70b/gov_report.slurm"
sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/pivot/pivot/b256/k7/llama32_1b_to_llama33_70b/gov_report.slurm
echo "sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/pivot/pivot/b256/k7/vicuna_68m_to_vicuna_13b_v13/gov_report.slurm"
sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/pivot/pivot/b256/k7/vicuna_68m_to_vicuna_13b_v13/gov_report.slurm
echo "sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/pivot/pivot_spechive/b256/k7/llama32_1b_to_llama33_70b/gov_report.slurm"
sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/pivot/pivot_spechive/b256/k7/llama32_1b_to_llama33_70b/gov_report.slurm
echo "sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/pivot/pivot_spechive/b256/k7/vicuna_68m_to_vicuna_13b_v13/gov_report.slurm"
sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/pivot/pivot_spechive/b256/k7/vicuna_68m_to_vicuna_13b_v13/gov_report.slurm
echo "sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/speculative/b256/k9/llama32_1b_to_llama33_70b/gov_report.slurm"
sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/speculative/b256/k9/llama32_1b_to_llama33_70b/gov_report.slurm
echo "sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/speculative/b256/k9/vicuna_68m_to_vicuna_13b_v13/gov_report.slurm"
sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/speculative/b256/k9/vicuna_68m_to_vicuna_13b_v13/gov_report.slurm
echo "sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/magicdec/b256/k9/llama32_1b_to_llama33_70b/gov_report.slurm"
sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/magicdec/b256/k9/llama32_1b_to_llama33_70b/gov_report.slurm
echo "sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/magicdec/b256/k9/vicuna_68m_to_vicuna_13b_v13/gov_report.slurm"
sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/magicdec/b256/k9/vicuna_68m_to_vicuna_13b_v13/gov_report.slurm
echo "sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/tetris/b256/k9/llama32_1b_to_llama33_70b/gov_report.slurm"
sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/tetris/b256/k9/llama32_1b_to_llama33_70b/gov_report.slurm
echo "sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/tetris/b256/k9/vicuna_68m_to_vicuna_13b_v13/gov_report.slurm"
sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/tetris/b256/k9/vicuna_68m_to_vicuna_13b_v13/gov_report.slurm
echo "sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/pivot/pivot/b256/k9/llama32_1b_to_llama33_70b/gov_report.slurm"
sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/pivot/pivot/b256/k9/llama32_1b_to_llama33_70b/gov_report.slurm
echo "sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/pivot/pivot/b256/k9/vicuna_68m_to_vicuna_13b_v13/gov_report.slurm"
sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/pivot/pivot/b256/k9/vicuna_68m_to_vicuna_13b_v13/gov_report.slurm
echo "sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/pivot/pivot_spechive/b256/k9/llama32_1b_to_llama33_70b/gov_report.slurm"
sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/pivot/pivot_spechive/b256/k9/llama32_1b_to_llama33_70b/gov_report.slurm
echo "sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/pivot/pivot_spechive/b256/k9/vicuna_68m_to_vicuna_13b_v13/gov_report.slurm"
sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/pivot/pivot_spechive/b256/k9/vicuna_68m_to_vicuna_13b_v13/gov_report.slurm
echo "sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/speculative/b256/k11/llama32_1b_to_llama33_70b/gov_report.slurm"
sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/speculative/b256/k11/llama32_1b_to_llama33_70b/gov_report.slurm
echo "sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/speculative/b256/k11/vicuna_68m_to_vicuna_13b_v13/gov_report.slurm"
sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/speculative/b256/k11/vicuna_68m_to_vicuna_13b_v13/gov_report.slurm
echo "sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/magicdec/b256/k11/llama32_1b_to_llama33_70b/gov_report.slurm"
sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/magicdec/b256/k11/llama32_1b_to_llama33_70b/gov_report.slurm
echo "sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/magicdec/b256/k11/vicuna_68m_to_vicuna_13b_v13/gov_report.slurm"
sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/magicdec/b256/k11/vicuna_68m_to_vicuna_13b_v13/gov_report.slurm
echo "sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/tetris/b256/k11/llama32_1b_to_llama33_70b/gov_report.slurm"
sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/tetris/b256/k11/llama32_1b_to_llama33_70b/gov_report.slurm
echo "sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/tetris/b256/k11/vicuna_68m_to_vicuna_13b_v13/gov_report.slurm"
sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/tetris/b256/k11/vicuna_68m_to_vicuna_13b_v13/gov_report.slurm
echo "sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/pivot/pivot/b256/k11/llama32_1b_to_llama33_70b/gov_report.slurm"
sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/pivot/pivot/b256/k11/llama32_1b_to_llama33_70b/gov_report.slurm
echo "sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/pivot/pivot/b256/k11/vicuna_68m_to_vicuna_13b_v13/gov_report.slurm"
sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/pivot/pivot/b256/k11/vicuna_68m_to_vicuna_13b_v13/gov_report.slurm
echo "sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/pivot/pivot_spechive/b256/k11/llama32_1b_to_llama33_70b/gov_report.slurm"
sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/pivot/pivot_spechive/b256/k11/llama32_1b_to_llama33_70b/gov_report.slurm
echo "sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/pivot/pivot_spechive/b256/k11/vicuna_68m_to_vicuna_13b_v13/gov_report.slurm"
sbatch /Users/junsu/EECE571/vllm-nips26/scripts/jobs/spec_decode/pivot/pivot_spechive/b256/k11/vicuna_68m_to_vicuna_13b_v13/gov_report.slurm
