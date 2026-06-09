#!/bin/bash
#SBATCH --job-name=vis_basins
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=05:30:00
#SBATCH --output=/vol/bitbucket/mz325/individual_project/logs/vis_basins_%j.out
#SBATCH --error=/vol/bitbucket/mz325/individual_project/logs/vis_basins_%j.err

echo "=========================================="
echo "Visualize Convergence Basins — Greedy Subsets"
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

mkdir -p vis_results/forward_greedy_bqs
mkdir -p logs

echo ""
echo "--- Generating 3D basin plots for best 2 subsets ---"
echo "  Subset 1: [6, 28, 34, 62, 12, 54, 3]  BQS=0.7402"
echo "  Subset 2: [19, 6, 28, 62, 52, 54]      BQS=0.6990"
echo "  Frame: 306  |  Conditions: Clean / +30% / +50%"
echo ""

python visualize_subset_basins.py

echo ""
echo "=========================================="
echo "Done at $(date)"
echo "Output: vis_results/forward_greedy_bqs/"
echo "=========================================="