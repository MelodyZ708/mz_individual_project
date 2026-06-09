#!/bin/bash
#SBATCH --job-name=vis_filter_weights
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=05:15:00
#SBATCH --output=/vol/bitbucket/mz325/individual_project/logs/vis_filter_weights_%j.out
#SBATCH --error=/vol/bitbucket/mz325/individual_project/logs/vis_filter_weights_%j.err

echo "=========================================="
echo "Conv1 Filter Weights Visualisation"
echo "Channels: [06, 28, 34, 62]"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "=========================================="

eval "$(/vol/bitbucket/mz325/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate /vol/bitbucket/mz325/envs/como

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

cd /vol/bitbucket/mz325/individual_project
mkdir -p vis_results/forward_greedy_bqs/analysis

python vis_filter_weights.py

echo ""
echo "Done at $(date)"
echo "Output: vis_results/forward_greedy_bqs/analysis/filter_weights_optimal4.png"
echo "=========================================="