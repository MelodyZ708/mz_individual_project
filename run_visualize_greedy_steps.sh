#!/bin/bash
#SBATCH --job-name=vis_steps
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/vol/bitbucket/mz325/individual_project/logs/vis_steps_%j.out
#SBATCH --error=/vol/bitbucket/mz325/individual_project/logs/vis_steps_%j.err

echo "=========================================="
echo "Visualize Greedy Steps — Per-Step Basin + BQS Components"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "=========================================="

eval "$(/vol/bitbucket/mz325/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate /vol/bitbucket/mz325/envs/como

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /vol/bitbucket/mz325/individual_project

mkdir -p vis_results/forward_greedy_bqs/step_vis/beam1
mkdir -p vis_results/forward_greedy_bqs/step_vis/beam2
mkdir -p logs

echo ""
echo "--- Generating per-step basin + BQS component figures ---"
echo "  Beam 1 (seed=ch6):  7 steps -> 7 figures"
echo "  Beam 2 (seed=ch19): 6 steps -> 6 figures"
echo "  Output: vis_results/forward_greedy_bqs/step_vis/"
echo ""

python visualize_greedy_steps.py

echo ""
echo "=========================================="
echo "Done at $(date)"
echo "Output: vis_results/forward_greedy_bqs/step_vis/"
echo "=========================================="