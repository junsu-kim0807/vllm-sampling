#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/scratch/junsuk87/vllm-sampling"

echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/speculative/b256/k7/llama32_1b_to_llama33_70b/aime25.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/speculative/b256/k7/llama32_1b_to_llama33_70b/aime25.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/magicdec/b256/k7/llama32_1b_to_llama33_70b/aime25.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/magicdec/b256/k7/llama32_1b_to_llama33_70b/aime25.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/tetris/b256/k7/llama32_1b_to_llama33_70b/aime25.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/tetris/b256/k7/llama32_1b_to_llama33_70b/aime25.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot/b256/k7/llama32_1b_to_llama33_70b/aime25.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot/b256/k7/llama32_1b_to_llama33_70b/aime25.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot_spechive/b256/k7/llama32_1b_to_llama33_70b/aime25.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot_spechive/b256/k7/llama32_1b_to_llama33_70b/aime25.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/speculative/b256/k7/llama32_1b_to_llama33_70b/alpaca.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/speculative/b256/k7/llama32_1b_to_llama33_70b/alpaca.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/magicdec/b256/k7/llama32_1b_to_llama33_70b/alpaca.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/magicdec/b256/k7/llama32_1b_to_llama33_70b/alpaca.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/tetris/b256/k7/llama32_1b_to_llama33_70b/alpaca.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/tetris/b256/k7/llama32_1b_to_llama33_70b/alpaca.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot/b256/k7/llama32_1b_to_llama33_70b/alpaca.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot/b256/k7/llama32_1b_to_llama33_70b/alpaca.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot_spechive/b256/k7/llama32_1b_to_llama33_70b/alpaca.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot_spechive/b256/k7/llama32_1b_to_llama33_70b/alpaca.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/eagle3/b256/k7/eagle3_llama33_70b/alpaca.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/eagle3/b256/k7/eagle3_llama33_70b/alpaca.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/speculative/b256/k11/llama32_1b_to_llama33_70b/aime25.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/speculative/b256/k11/llama32_1b_to_llama33_70b/aime25.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/magicdec/b256/k11/llama32_1b_to_llama33_70b/aime25.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/magicdec/b256/k11/llama32_1b_to_llama33_70b/aime25.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/tetris/b256/k11/llama32_1b_to_llama33_70b/aime25.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/tetris/b256/k11/llama32_1b_to_llama33_70b/aime25.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot/b256/k11/llama32_1b_to_llama33_70b/aime25.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot/b256/k11/llama32_1b_to_llama33_70b/aime25.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot_spechive/b256/k11/llama32_1b_to_llama33_70b/aime25.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot_spechive/b256/k11/llama32_1b_to_llama33_70b/aime25.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/speculative/b256/k11/llama32_1b_to_llama33_70b/alpaca.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/speculative/b256/k11/llama32_1b_to_llama33_70b/alpaca.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/magicdec/b256/k11/llama32_1b_to_llama33_70b/alpaca.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/magicdec/b256/k11/llama32_1b_to_llama33_70b/alpaca.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/tetris/b256/k11/llama32_1b_to_llama33_70b/alpaca.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/tetris/b256/k11/llama32_1b_to_llama33_70b/alpaca.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot/b256/k11/llama32_1b_to_llama33_70b/alpaca.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot/b256/k11/llama32_1b_to_llama33_70b/alpaca.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot_spechive/b256/k11/llama32_1b_to_llama33_70b/alpaca.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot_spechive/b256/k11/llama32_1b_to_llama33_70b/alpaca.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/eagle3/b256/k11/eagle3_llama33_70b/alpaca.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/eagle3/b256/k11/eagle3_llama33_70b/alpaca.slurm
