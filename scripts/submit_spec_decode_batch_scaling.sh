#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/Users/junsu/StarKV/vllm-sampling"

echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_llama33_70b/gov_report_b1.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_llama33_70b/gov_report_b1.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_qwen30b_a3b/gov_report_b1.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_qwen30b_a3b/gov_report_b1.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_deepseekcoder_33b/gov_report_b1.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_deepseekcoder_33b/gov_report_b1.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_1b_to_llama33_70b/gov_report_b1.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_1b_to_llama33_70b/gov_report_b1.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_3b_to_llama33_70b/gov_report_b1.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_3b_to_llama33_70b/gov_report_b1.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_1p3b_to_33b/gov_report_b1.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_1p3b_to_33b/gov_report_b1.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_6p7b_to_33b/gov_report_b1.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_6p7b_to_33b/gov_report_b1.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b/gov_report_b1.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b/gov_report_b1.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_4b_to_qwen3_30b_a3b/gov_report_b1.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_4b_to_qwen3_30b_a3b/gov_report_b1.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_4b_instruct_2507/gov_report_b1.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_4b_instruct_2507/gov_report_b1.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_30b_a3b_instruct_2507/gov_report_b1.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_30b_a3b_instruct_2507/gov_report_b1.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_4b_instruct_2507/gov_report_b1.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_4b_instruct_2507/gov_report_b1.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b_instruct_2507/gov_report_b1.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b_instruct_2507/gov_report_b1.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_llama33_70b/gov_report_b1.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_llama33_70b/gov_report_b1.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_qwen30b_a3b/gov_report_b1.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_qwen30b_a3b/gov_report_b1.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_llama33_70b/qmsum_b1.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_llama33_70b/qmsum_b1.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_qwen30b_a3b/qmsum_b1.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_qwen30b_a3b/qmsum_b1.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_deepseekcoder_33b/qmsum_b1.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_deepseekcoder_33b/qmsum_b1.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_1b_to_llama33_70b/qmsum_b1.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_1b_to_llama33_70b/qmsum_b1.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_3b_to_llama33_70b/qmsum_b1.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_3b_to_llama33_70b/qmsum_b1.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_1p3b_to_33b/qmsum_b1.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_1p3b_to_33b/qmsum_b1.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_6p7b_to_33b/qmsum_b1.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_6p7b_to_33b/qmsum_b1.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b/qmsum_b1.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b/qmsum_b1.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_4b_to_qwen3_30b_a3b/qmsum_b1.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_4b_to_qwen3_30b_a3b/qmsum_b1.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_4b_instruct_2507/qmsum_b1.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_4b_instruct_2507/qmsum_b1.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_30b_a3b_instruct_2507/qmsum_b1.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_30b_a3b_instruct_2507/qmsum_b1.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_4b_instruct_2507/qmsum_b1.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_4b_instruct_2507/qmsum_b1.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b_instruct_2507/qmsum_b1.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b_instruct_2507/qmsum_b1.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_llama33_70b/qmsum_b1.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_llama33_70b/qmsum_b1.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_qwen30b_a3b/qmsum_b1.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_qwen30b_a3b/qmsum_b1.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_llama33_70b/gov_report_b4.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_llama33_70b/gov_report_b4.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_qwen30b_a3b/gov_report_b4.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_qwen30b_a3b/gov_report_b4.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_deepseekcoder_33b/gov_report_b4.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_deepseekcoder_33b/gov_report_b4.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_1b_to_llama33_70b/gov_report_b4.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_1b_to_llama33_70b/gov_report_b4.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_3b_to_llama33_70b/gov_report_b4.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_3b_to_llama33_70b/gov_report_b4.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_1p3b_to_33b/gov_report_b4.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_1p3b_to_33b/gov_report_b4.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_6p7b_to_33b/gov_report_b4.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_6p7b_to_33b/gov_report_b4.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b/gov_report_b4.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b/gov_report_b4.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_4b_to_qwen3_30b_a3b/gov_report_b4.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_4b_to_qwen3_30b_a3b/gov_report_b4.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_4b_instruct_2507/gov_report_b4.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_4b_instruct_2507/gov_report_b4.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_30b_a3b_instruct_2507/gov_report_b4.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_30b_a3b_instruct_2507/gov_report_b4.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_4b_instruct_2507/gov_report_b4.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_4b_instruct_2507/gov_report_b4.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b_instruct_2507/gov_report_b4.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b_instruct_2507/gov_report_b4.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_llama33_70b/gov_report_b4.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_llama33_70b/gov_report_b4.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_qwen30b_a3b/gov_report_b4.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_qwen30b_a3b/gov_report_b4.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_llama33_70b/qmsum_b4.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_llama33_70b/qmsum_b4.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_qwen30b_a3b/qmsum_b4.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_qwen30b_a3b/qmsum_b4.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_deepseekcoder_33b/qmsum_b4.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_deepseekcoder_33b/qmsum_b4.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_1b_to_llama33_70b/qmsum_b4.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_1b_to_llama33_70b/qmsum_b4.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_3b_to_llama33_70b/qmsum_b4.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_3b_to_llama33_70b/qmsum_b4.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_1p3b_to_33b/qmsum_b4.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_1p3b_to_33b/qmsum_b4.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_6p7b_to_33b/qmsum_b4.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_6p7b_to_33b/qmsum_b4.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b/qmsum_b4.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b/qmsum_b4.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_4b_to_qwen3_30b_a3b/qmsum_b4.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_4b_to_qwen3_30b_a3b/qmsum_b4.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_4b_instruct_2507/qmsum_b4.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_4b_instruct_2507/qmsum_b4.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_30b_a3b_instruct_2507/qmsum_b4.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_30b_a3b_instruct_2507/qmsum_b4.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_4b_instruct_2507/qmsum_b4.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_4b_instruct_2507/qmsum_b4.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b_instruct_2507/qmsum_b4.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b_instruct_2507/qmsum_b4.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_llama33_70b/qmsum_b4.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_llama33_70b/qmsum_b4.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_qwen30b_a3b/qmsum_b4.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_qwen30b_a3b/qmsum_b4.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_llama33_70b/gov_report_b16.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_llama33_70b/gov_report_b16.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_qwen30b_a3b/gov_report_b16.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_qwen30b_a3b/gov_report_b16.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_deepseekcoder_33b/gov_report_b16.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_deepseekcoder_33b/gov_report_b16.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_1b_to_llama33_70b/gov_report_b16.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_1b_to_llama33_70b/gov_report_b16.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_3b_to_llama33_70b/gov_report_b16.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_3b_to_llama33_70b/gov_report_b16.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_1p3b_to_33b/gov_report_b16.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_1p3b_to_33b/gov_report_b16.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_6p7b_to_33b/gov_report_b16.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_6p7b_to_33b/gov_report_b16.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b/gov_report_b16.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b/gov_report_b16.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_4b_to_qwen3_30b_a3b/gov_report_b16.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_4b_to_qwen3_30b_a3b/gov_report_b16.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_4b_instruct_2507/gov_report_b16.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_4b_instruct_2507/gov_report_b16.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_30b_a3b_instruct_2507/gov_report_b16.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_30b_a3b_instruct_2507/gov_report_b16.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_4b_instruct_2507/gov_report_b16.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_4b_instruct_2507/gov_report_b16.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b_instruct_2507/gov_report_b16.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b_instruct_2507/gov_report_b16.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_llama33_70b/gov_report_b16.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_llama33_70b/gov_report_b16.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_qwen30b_a3b/gov_report_b16.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_qwen30b_a3b/gov_report_b16.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_llama33_70b/qmsum_b16.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_llama33_70b/qmsum_b16.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_qwen30b_a3b/qmsum_b16.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_qwen30b_a3b/qmsum_b16.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_deepseekcoder_33b/qmsum_b16.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_deepseekcoder_33b/qmsum_b16.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_1b_to_llama33_70b/qmsum_b16.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_1b_to_llama33_70b/qmsum_b16.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_3b_to_llama33_70b/qmsum_b16.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_3b_to_llama33_70b/qmsum_b16.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_1p3b_to_33b/qmsum_b16.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_1p3b_to_33b/qmsum_b16.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_6p7b_to_33b/qmsum_b16.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_6p7b_to_33b/qmsum_b16.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b/qmsum_b16.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b/qmsum_b16.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_4b_to_qwen3_30b_a3b/qmsum_b16.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_4b_to_qwen3_30b_a3b/qmsum_b16.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_4b_instruct_2507/qmsum_b16.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_4b_instruct_2507/qmsum_b16.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_30b_a3b_instruct_2507/qmsum_b16.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_30b_a3b_instruct_2507/qmsum_b16.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_4b_instruct_2507/qmsum_b16.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_4b_instruct_2507/qmsum_b16.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b_instruct_2507/qmsum_b16.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b_instruct_2507/qmsum_b16.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_llama33_70b/qmsum_b16.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_llama33_70b/qmsum_b16.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_qwen30b_a3b/qmsum_b16.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_qwen30b_a3b/qmsum_b16.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_llama33_70b/gov_report_b64.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_llama33_70b/gov_report_b64.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_qwen30b_a3b/gov_report_b64.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_qwen30b_a3b/gov_report_b64.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_deepseekcoder_33b/gov_report_b64.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_deepseekcoder_33b/gov_report_b64.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_1b_to_llama33_70b/gov_report_b64.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_1b_to_llama33_70b/gov_report_b64.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_3b_to_llama33_70b/gov_report_b64.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_3b_to_llama33_70b/gov_report_b64.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_1p3b_to_33b/gov_report_b64.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_1p3b_to_33b/gov_report_b64.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_6p7b_to_33b/gov_report_b64.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_6p7b_to_33b/gov_report_b64.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b/gov_report_b64.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b/gov_report_b64.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_4b_to_qwen3_30b_a3b/gov_report_b64.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_4b_to_qwen3_30b_a3b/gov_report_b64.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_4b_instruct_2507/gov_report_b64.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_4b_instruct_2507/gov_report_b64.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_30b_a3b_instruct_2507/gov_report_b64.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_30b_a3b_instruct_2507/gov_report_b64.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_4b_instruct_2507/gov_report_b64.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_4b_instruct_2507/gov_report_b64.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b_instruct_2507/gov_report_b64.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b_instruct_2507/gov_report_b64.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_llama33_70b/gov_report_b64.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_llama33_70b/gov_report_b64.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_qwen30b_a3b/gov_report_b64.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_qwen30b_a3b/gov_report_b64.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_llama33_70b/qmsum_b64.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_llama33_70b/qmsum_b64.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_qwen30b_a3b/qmsum_b64.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_qwen30b_a3b/qmsum_b64.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_deepseekcoder_33b/qmsum_b64.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_deepseekcoder_33b/qmsum_b64.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_1b_to_llama33_70b/qmsum_b64.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_1b_to_llama33_70b/qmsum_b64.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_3b_to_llama33_70b/qmsum_b64.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_3b_to_llama33_70b/qmsum_b64.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_1p3b_to_33b/qmsum_b64.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_1p3b_to_33b/qmsum_b64.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_6p7b_to_33b/qmsum_b64.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_6p7b_to_33b/qmsum_b64.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b/qmsum_b64.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b/qmsum_b64.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_4b_to_qwen3_30b_a3b/qmsum_b64.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_4b_to_qwen3_30b_a3b/qmsum_b64.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_4b_instruct_2507/qmsum_b64.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_4b_instruct_2507/qmsum_b64.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_30b_a3b_instruct_2507/qmsum_b64.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_30b_a3b_instruct_2507/qmsum_b64.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_4b_instruct_2507/qmsum_b64.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_4b_instruct_2507/qmsum_b64.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b_instruct_2507/qmsum_b64.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b_instruct_2507/qmsum_b64.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_llama33_70b/qmsum_b64.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_llama33_70b/qmsum_b64.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_qwen30b_a3b/qmsum_b64.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_qwen30b_a3b/qmsum_b64.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_llama33_70b/gov_report_b256.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_llama33_70b/gov_report_b256.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_qwen30b_a3b/gov_report_b256.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_qwen30b_a3b/gov_report_b256.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_deepseekcoder_33b/gov_report_b256.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_deepseekcoder_33b/gov_report_b256.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_1b_to_llama33_70b/gov_report_b256.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_1b_to_llama33_70b/gov_report_b256.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_3b_to_llama33_70b/gov_report_b256.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_3b_to_llama33_70b/gov_report_b256.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_1p3b_to_33b/gov_report_b256.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_1p3b_to_33b/gov_report_b256.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_6p7b_to_33b/gov_report_b256.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_6p7b_to_33b/gov_report_b256.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b/gov_report_b256.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b/gov_report_b256.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_4b_to_qwen3_30b_a3b/gov_report_b256.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_4b_to_qwen3_30b_a3b/gov_report_b256.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_4b_instruct_2507/gov_report_b256.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_4b_instruct_2507/gov_report_b256.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_30b_a3b_instruct_2507/gov_report_b256.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_30b_a3b_instruct_2507/gov_report_b256.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_4b_instruct_2507/gov_report_b256.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_4b_instruct_2507/gov_report_b256.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b_instruct_2507/gov_report_b256.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b_instruct_2507/gov_report_b256.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_llama33_70b/gov_report_b256.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_llama33_70b/gov_report_b256.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_qwen30b_a3b/gov_report_b256.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_qwen30b_a3b/gov_report_b256.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_llama33_70b/qmsum_b256.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_llama33_70b/qmsum_b256.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_qwen30b_a3b/qmsum_b256.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_qwen30b_a3b/qmsum_b256.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_deepseekcoder_33b/qmsum_b256.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_deepseekcoder_33b/qmsum_b256.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_1b_to_llama33_70b/qmsum_b256.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_1b_to_llama33_70b/qmsum_b256.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_3b_to_llama33_70b/qmsum_b256.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_3b_to_llama33_70b/qmsum_b256.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_1p3b_to_33b/qmsum_b256.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_1p3b_to_33b/qmsum_b256.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_6p7b_to_33b/qmsum_b256.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_6p7b_to_33b/qmsum_b256.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b/qmsum_b256.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b/qmsum_b256.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_4b_to_qwen3_30b_a3b/qmsum_b256.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_4b_to_qwen3_30b_a3b/qmsum_b256.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_4b_instruct_2507/qmsum_b256.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_4b_instruct_2507/qmsum_b256.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_30b_a3b_instruct_2507/qmsum_b256.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_30b_a3b_instruct_2507/qmsum_b256.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_4b_instruct_2507/qmsum_b256.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_4b_instruct_2507/qmsum_b256.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b_instruct_2507/qmsum_b256.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b_instruct_2507/qmsum_b256.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_llama33_70b/qmsum_b256.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_llama33_70b/qmsum_b256.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_qwen30b_a3b/qmsum_b256.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_qwen30b_a3b/qmsum_b256.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_llama33_70b/gov_report_b512.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_llama33_70b/gov_report_b512.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_qwen30b_a3b/gov_report_b512.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_qwen30b_a3b/gov_report_b512.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_deepseekcoder_33b/gov_report_b512.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_deepseekcoder_33b/gov_report_b512.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_1b_to_llama33_70b/gov_report_b512.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_1b_to_llama33_70b/gov_report_b512.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_3b_to_llama33_70b/gov_report_b512.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_3b_to_llama33_70b/gov_report_b512.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_1p3b_to_33b/gov_report_b512.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_1p3b_to_33b/gov_report_b512.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_6p7b_to_33b/gov_report_b512.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_6p7b_to_33b/gov_report_b512.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b/gov_report_b512.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b/gov_report_b512.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_4b_to_qwen3_30b_a3b/gov_report_b512.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_4b_to_qwen3_30b_a3b/gov_report_b512.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_4b_instruct_2507/gov_report_b512.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_4b_instruct_2507/gov_report_b512.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_30b_a3b_instruct_2507/gov_report_b512.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_30b_a3b_instruct_2507/gov_report_b512.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_4b_instruct_2507/gov_report_b512.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_4b_instruct_2507/gov_report_b512.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b_instruct_2507/gov_report_b512.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b_instruct_2507/gov_report_b512.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_llama33_70b/gov_report_b512.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_llama33_70b/gov_report_b512.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_qwen30b_a3b/gov_report_b512.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_qwen30b_a3b/gov_report_b512.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_llama33_70b/qmsum_b512.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_llama33_70b/qmsum_b512.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_qwen30b_a3b/qmsum_b512.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_qwen30b_a3b/qmsum_b512.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_deepseekcoder_33b/qmsum_b512.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_deepseekcoder_33b/qmsum_b512.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_1b_to_llama33_70b/qmsum_b512.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_1b_to_llama33_70b/qmsum_b512.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_3b_to_llama33_70b/qmsum_b512.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_3b_to_llama33_70b/qmsum_b512.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_1p3b_to_33b/qmsum_b512.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_1p3b_to_33b/qmsum_b512.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_6p7b_to_33b/qmsum_b512.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_6p7b_to_33b/qmsum_b512.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b/qmsum_b512.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b/qmsum_b512.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_4b_to_qwen3_30b_a3b/qmsum_b512.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_4b_to_qwen3_30b_a3b/qmsum_b512.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_4b_instruct_2507/qmsum_b512.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_4b_instruct_2507/qmsum_b512.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_30b_a3b_instruct_2507/qmsum_b512.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_30b_a3b_instruct_2507/qmsum_b512.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_4b_instruct_2507/qmsum_b512.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_4b_instruct_2507/qmsum_b512.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b_instruct_2507/qmsum_b512.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b_instruct_2507/qmsum_b512.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_llama33_70b/qmsum_b512.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_llama33_70b/qmsum_b512.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_qwen30b_a3b/qmsum_b512.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_qwen30b_a3b/qmsum_b512.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_llama33_70b/gov_report_b1024.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_llama33_70b/gov_report_b1024.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_qwen30b_a3b/gov_report_b1024.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_qwen30b_a3b/gov_report_b1024.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_deepseekcoder_33b/gov_report_b1024.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_deepseekcoder_33b/gov_report_b1024.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_1b_to_llama33_70b/gov_report_b1024.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_1b_to_llama33_70b/gov_report_b1024.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_3b_to_llama33_70b/gov_report_b1024.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_3b_to_llama33_70b/gov_report_b1024.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_1p3b_to_33b/gov_report_b1024.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_1p3b_to_33b/gov_report_b1024.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_6p7b_to_33b/gov_report_b1024.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_6p7b_to_33b/gov_report_b1024.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b/gov_report_b1024.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b/gov_report_b1024.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_4b_to_qwen3_30b_a3b/gov_report_b1024.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_4b_to_qwen3_30b_a3b/gov_report_b1024.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_4b_instruct_2507/gov_report_b1024.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_4b_instruct_2507/gov_report_b1024.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_30b_a3b_instruct_2507/gov_report_b1024.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_30b_a3b_instruct_2507/gov_report_b1024.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_4b_instruct_2507/gov_report_b1024.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_4b_instruct_2507/gov_report_b1024.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b_instruct_2507/gov_report_b1024.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b_instruct_2507/gov_report_b1024.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_llama33_70b/gov_report_b1024.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_llama33_70b/gov_report_b1024.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_qwen30b_a3b/gov_report_b1024.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_qwen30b_a3b/gov_report_b1024.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_llama33_70b/qmsum_b1024.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_llama33_70b/qmsum_b1024.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_qwen30b_a3b/qmsum_b1024.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_qwen30b_a3b/qmsum_b1024.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_deepseekcoder_33b/qmsum_b1024.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/ar_deepseekcoder_33b/qmsum_b1024.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_1b_to_llama33_70b/qmsum_b1024.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_1b_to_llama33_70b/qmsum_b1024.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_3b_to_llama33_70b/qmsum_b1024.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/llama32_3b_to_llama33_70b/qmsum_b1024.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_1p3b_to_33b/qmsum_b1024.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_1p3b_to_33b/qmsum_b1024.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_6p7b_to_33b/qmsum_b1024.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/deepseekcoder_6p7b_to_33b/qmsum_b1024.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b/qmsum_b1024.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b/qmsum_b1024.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_4b_to_qwen3_30b_a3b/qmsum_b1024.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_4b_to_qwen3_30b_a3b/qmsum_b1024.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_4b_instruct_2507/qmsum_b1024.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_4b_instruct_2507/qmsum_b1024.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_30b_a3b_instruct_2507/qmsum_b1024.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen25_0p5b_to_qwen3_30b_a3b_instruct_2507/qmsum_b1024.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_4b_instruct_2507/qmsum_b1024.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_4b_instruct_2507/qmsum_b1024.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b_instruct_2507/qmsum_b1024.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/qwen3_0p6b_to_qwen3_30b_a3b_instruct_2507/qmsum_b1024.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_llama33_70b/qmsum_b1024.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_llama33_70b/qmsum_b1024.slurm
echo "sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_qwen30b_a3b/qmsum_b1024.slurm"
sbatch /Users/junsu/StarKV/vllm-sampling/scripts/jobs/spec_decode/eagle3_qwen30b_a3b/qmsum_b1024.slurm
