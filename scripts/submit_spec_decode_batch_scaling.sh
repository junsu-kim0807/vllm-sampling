#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/scratch/jhwoo36/vllm-sampling"

echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/ar_llama33_70b/codeelo_b1.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/ar_llama33_70b/codeelo_b1.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/ar_qwen30b_a3b/codeelo_b1.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/ar_qwen30b_a3b/codeelo_b1.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/llama32_1b_to_llama33_70b/codeelo_b1.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/llama32_1b_to_llama33_70b/codeelo_b1.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/llama32_3b_to_llama33_70b/codeelo_b1.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/llama32_3b_to_llama33_70b/codeelo_b1.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_1p3b_to_33b/codeelo_b1.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_1p3b_to_33b/codeelo_b1.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_6p7b_to_33b/codeelo_b1.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_6p7b_to_33b/codeelo_b1.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b/codeelo_b1.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b/codeelo_b1.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/qwen3_4b_to_qwen3_30b_a3b/codeelo_b1.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/qwen3_4b_to_qwen3_30b_a3b/codeelo_b1.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/eagle3_llama33_70b/codeelo_b1.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/eagle3_llama33_70b/codeelo_b1.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/eagle3_qwen30b_a3b/codeelo_b1.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/eagle3_qwen30b_a3b/codeelo_b1.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/ar_llama33_70b/codeelo_b4.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/ar_llama33_70b/codeelo_b4.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/ar_qwen30b_a3b/codeelo_b4.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/ar_qwen30b_a3b/codeelo_b4.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/llama32_1b_to_llama33_70b/codeelo_b4.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/llama32_1b_to_llama33_70b/codeelo_b4.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/llama32_3b_to_llama33_70b/codeelo_b4.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/llama32_3b_to_llama33_70b/codeelo_b4.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_1p3b_to_33b/codeelo_b4.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_1p3b_to_33b/codeelo_b4.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_6p7b_to_33b/codeelo_b4.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_6p7b_to_33b/codeelo_b4.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b/codeelo_b4.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b/codeelo_b4.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/qwen3_4b_to_qwen3_30b_a3b/codeelo_b4.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/qwen3_4b_to_qwen3_30b_a3b/codeelo_b4.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/eagle3_llama33_70b/codeelo_b4.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/eagle3_llama33_70b/codeelo_b4.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/eagle3_qwen30b_a3b/codeelo_b4.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/eagle3_qwen30b_a3b/codeelo_b4.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/ar_llama33_70b/codeelo_b16.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/ar_llama33_70b/codeelo_b16.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/ar_qwen30b_a3b/codeelo_b16.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/ar_qwen30b_a3b/codeelo_b16.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/llama32_1b_to_llama33_70b/codeelo_b16.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/llama32_1b_to_llama33_70b/codeelo_b16.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/llama32_3b_to_llama33_70b/codeelo_b16.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/llama32_3b_to_llama33_70b/codeelo_b16.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_1p3b_to_33b/codeelo_b16.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_1p3b_to_33b/codeelo_b16.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_6p7b_to_33b/codeelo_b16.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_6p7b_to_33b/codeelo_b16.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b/codeelo_b16.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b/codeelo_b16.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/qwen3_4b_to_qwen3_30b_a3b/codeelo_b16.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/qwen3_4b_to_qwen3_30b_a3b/codeelo_b16.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/eagle3_llama33_70b/codeelo_b16.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/eagle3_llama33_70b/codeelo_b16.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/eagle3_qwen30b_a3b/codeelo_b16.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/eagle3_qwen30b_a3b/codeelo_b16.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/ar_llama33_70b/codeelo_b64.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/ar_llama33_70b/codeelo_b64.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/ar_qwen30b_a3b/codeelo_b64.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/ar_qwen30b_a3b/codeelo_b64.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/llama32_1b_to_llama33_70b/codeelo_b64.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/llama32_1b_to_llama33_70b/codeelo_b64.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/llama32_3b_to_llama33_70b/codeelo_b64.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/llama32_3b_to_llama33_70b/codeelo_b64.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_1p3b_to_33b/codeelo_b64.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_1p3b_to_33b/codeelo_b64.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_6p7b_to_33b/codeelo_b64.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_6p7b_to_33b/codeelo_b64.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b/codeelo_b64.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b/codeelo_b64.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/qwen3_4b_to_qwen3_30b_a3b/codeelo_b64.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/qwen3_4b_to_qwen3_30b_a3b/codeelo_b64.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/eagle3_llama33_70b/codeelo_b64.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/eagle3_llama33_70b/codeelo_b64.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/eagle3_qwen30b_a3b/codeelo_b64.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/eagle3_qwen30b_a3b/codeelo_b64.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/ar_llama33_70b/codeelo_b256.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/ar_llama33_70b/codeelo_b256.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/ar_qwen30b_a3b/codeelo_b256.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/ar_qwen30b_a3b/codeelo_b256.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/llama32_1b_to_llama33_70b/codeelo_b256.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/llama32_1b_to_llama33_70b/codeelo_b256.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/llama32_3b_to_llama33_70b/codeelo_b256.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/llama32_3b_to_llama33_70b/codeelo_b256.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_1p3b_to_33b/codeelo_b256.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_1p3b_to_33b/codeelo_b256.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_6p7b_to_33b/codeelo_b256.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_6p7b_to_33b/codeelo_b256.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b/codeelo_b256.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b/codeelo_b256.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/qwen3_4b_to_qwen3_30b_a3b/codeelo_b256.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/qwen3_4b_to_qwen3_30b_a3b/codeelo_b256.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/eagle3_llama33_70b/codeelo_b256.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/eagle3_llama33_70b/codeelo_b256.slurm
echo "sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/eagle3_qwen30b_a3b/codeelo_b256.slurm"
sbatch /scratch/jhwoo36/vllm-sampling/scripts/jobs/spec_decode/eagle3_qwen30b_a3b/codeelo_b256.slurm
