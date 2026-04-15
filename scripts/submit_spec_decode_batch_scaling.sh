#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/scratch/junsuk87/vllm-sampling"

echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/speculative/b1/k3/llama32_1b_to_llama33_70b/aime25.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/speculative/b1/k3/llama32_1b_to_llama33_70b/aime25.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/magicdec/b1/k3/llama32_1b_to_llama33_70b/aime25.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/magicdec/b1/k3/llama32_1b_to_llama33_70b/aime25.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/tetris/b1/k3/llama32_1b_to_llama33_70b/aime25.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/tetris/b1/k3/llama32_1b_to_llama33_70b/aime25.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot/b1/k3/llama32_1b_to_llama33_70b/aime25.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot/b1/k3/llama32_1b_to_llama33_70b/aime25.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot_spechive/b1/k3/llama32_1b_to_llama33_70b/aime25.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot_spechive/b1/k3/llama32_1b_to_llama33_70b/aime25.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/speculative/b1/k3/llama32_1b_to_llama33_70b/alpaca.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/speculative/b1/k3/llama32_1b_to_llama33_70b/alpaca.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/magicdec/b1/k3/llama32_1b_to_llama33_70b/alpaca.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/magicdec/b1/k3/llama32_1b_to_llama33_70b/alpaca.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/tetris/b1/k3/llama32_1b_to_llama33_70b/alpaca.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/tetris/b1/k3/llama32_1b_to_llama33_70b/alpaca.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot/b1/k3/llama32_1b_to_llama33_70b/alpaca.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot/b1/k3/llama32_1b_to_llama33_70b/alpaca.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot_spechive/b1/k3/llama32_1b_to_llama33_70b/alpaca.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot_spechive/b1/k3/llama32_1b_to_llama33_70b/alpaca.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/eagle3/b1/k3/eagle3_llama33_70b/alpaca.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/eagle3/b1/k3/eagle3_llama33_70b/alpaca.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/speculative/b4/k3/llama32_1b_to_llama33_70b/aime25.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/speculative/b4/k3/llama32_1b_to_llama33_70b/aime25.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/magicdec/b4/k3/llama32_1b_to_llama33_70b/aime25.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/magicdec/b4/k3/llama32_1b_to_llama33_70b/aime25.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/tetris/b4/k3/llama32_1b_to_llama33_70b/aime25.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/tetris/b4/k3/llama32_1b_to_llama33_70b/aime25.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot/b4/k3/llama32_1b_to_llama33_70b/aime25.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot/b4/k3/llama32_1b_to_llama33_70b/aime25.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot_spechive/b4/k3/llama32_1b_to_llama33_70b/aime25.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot_spechive/b4/k3/llama32_1b_to_llama33_70b/aime25.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/speculative/b4/k3/llama32_1b_to_llama33_70b/alpaca.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/speculative/b4/k3/llama32_1b_to_llama33_70b/alpaca.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/magicdec/b4/k3/llama32_1b_to_llama33_70b/alpaca.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/magicdec/b4/k3/llama32_1b_to_llama33_70b/alpaca.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/tetris/b4/k3/llama32_1b_to_llama33_70b/alpaca.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/tetris/b4/k3/llama32_1b_to_llama33_70b/alpaca.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot/b4/k3/llama32_1b_to_llama33_70b/alpaca.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot/b4/k3/llama32_1b_to_llama33_70b/alpaca.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot_spechive/b4/k3/llama32_1b_to_llama33_70b/alpaca.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot_spechive/b4/k3/llama32_1b_to_llama33_70b/alpaca.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/eagle3/b4/k3/eagle3_llama33_70b/alpaca.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/eagle3/b4/k3/eagle3_llama33_70b/alpaca.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/speculative/b16/k3/llama32_1b_to_llama33_70b/aime25.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/speculative/b16/k3/llama32_1b_to_llama33_70b/aime25.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/magicdec/b16/k3/llama32_1b_to_llama33_70b/aime25.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/magicdec/b16/k3/llama32_1b_to_llama33_70b/aime25.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/tetris/b16/k3/llama32_1b_to_llama33_70b/aime25.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/tetris/b16/k3/llama32_1b_to_llama33_70b/aime25.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot/b16/k3/llama32_1b_to_llama33_70b/aime25.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot/b16/k3/llama32_1b_to_llama33_70b/aime25.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot_spechive/b16/k3/llama32_1b_to_llama33_70b/aime25.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot_spechive/b16/k3/llama32_1b_to_llama33_70b/aime25.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/speculative/b16/k3/llama32_1b_to_llama33_70b/alpaca.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/speculative/b16/k3/llama32_1b_to_llama33_70b/alpaca.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/magicdec/b16/k3/llama32_1b_to_llama33_70b/alpaca.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/magicdec/b16/k3/llama32_1b_to_llama33_70b/alpaca.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/tetris/b16/k3/llama32_1b_to_llama33_70b/alpaca.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/tetris/b16/k3/llama32_1b_to_llama33_70b/alpaca.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot/b16/k3/llama32_1b_to_llama33_70b/alpaca.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot/b16/k3/llama32_1b_to_llama33_70b/alpaca.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot_spechive/b16/k3/llama32_1b_to_llama33_70b/alpaca.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot_spechive/b16/k3/llama32_1b_to_llama33_70b/alpaca.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/eagle3/b16/k3/eagle3_llama33_70b/alpaca.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/eagle3/b16/k3/eagle3_llama33_70b/alpaca.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/speculative/b64/k3/llama32_1b_to_llama33_70b/aime25.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/speculative/b64/k3/llama32_1b_to_llama33_70b/aime25.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/magicdec/b64/k3/llama32_1b_to_llama33_70b/aime25.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/magicdec/b64/k3/llama32_1b_to_llama33_70b/aime25.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/tetris/b64/k3/llama32_1b_to_llama33_70b/aime25.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/tetris/b64/k3/llama32_1b_to_llama33_70b/aime25.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot/b64/k3/llama32_1b_to_llama33_70b/aime25.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot/b64/k3/llama32_1b_to_llama33_70b/aime25.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot_spechive/b64/k3/llama32_1b_to_llama33_70b/aime25.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot_spechive/b64/k3/llama32_1b_to_llama33_70b/aime25.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/speculative/b64/k3/llama32_1b_to_llama33_70b/alpaca.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/speculative/b64/k3/llama32_1b_to_llama33_70b/alpaca.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/magicdec/b64/k3/llama32_1b_to_llama33_70b/alpaca.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/magicdec/b64/k3/llama32_1b_to_llama33_70b/alpaca.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/tetris/b64/k3/llama32_1b_to_llama33_70b/alpaca.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/tetris/b64/k3/llama32_1b_to_llama33_70b/alpaca.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot/b64/k3/llama32_1b_to_llama33_70b/alpaca.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot/b64/k3/llama32_1b_to_llama33_70b/alpaca.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot_spechive/b64/k3/llama32_1b_to_llama33_70b/alpaca.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot_spechive/b64/k3/llama32_1b_to_llama33_70b/alpaca.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/eagle3/b64/k3/eagle3_llama33_70b/alpaca.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/eagle3/b64/k3/eagle3_llama33_70b/alpaca.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/speculative/b256/k3/llama32_1b_to_llama33_70b/aime25.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/speculative/b256/k3/llama32_1b_to_llama33_70b/aime25.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/magicdec/b256/k3/llama32_1b_to_llama33_70b/aime25.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/magicdec/b256/k3/llama32_1b_to_llama33_70b/aime25.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/tetris/b256/k3/llama32_1b_to_llama33_70b/aime25.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/tetris/b256/k3/llama32_1b_to_llama33_70b/aime25.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot/b256/k3/llama32_1b_to_llama33_70b/aime25.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot/b256/k3/llama32_1b_to_llama33_70b/aime25.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot_spechive/b256/k3/llama32_1b_to_llama33_70b/aime25.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot_spechive/b256/k3/llama32_1b_to_llama33_70b/aime25.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/speculative/b256/k3/llama32_1b_to_llama33_70b/alpaca.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/speculative/b256/k3/llama32_1b_to_llama33_70b/alpaca.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/magicdec/b256/k3/llama32_1b_to_llama33_70b/alpaca.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/magicdec/b256/k3/llama32_1b_to_llama33_70b/alpaca.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/tetris/b256/k3/llama32_1b_to_llama33_70b/alpaca.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/tetris/b256/k3/llama32_1b_to_llama33_70b/alpaca.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot/b256/k3/llama32_1b_to_llama33_70b/alpaca.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot/b256/k3/llama32_1b_to_llama33_70b/alpaca.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot_spechive/b256/k3/llama32_1b_to_llama33_70b/alpaca.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/pivot/pivot_spechive/b256/k3/llama32_1b_to_llama33_70b/alpaca.slurm
echo "sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/eagle3/b256/k3/eagle3_llama33_70b/alpaca.slurm"
sbatch /scratch/junsuk87/vllm-sampling/scripts/jobs/spec_decode/eagle3/b256/k3/eagle3_llama33_70b/alpaca.slurm
