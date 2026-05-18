#!/bin/bash
#SBATCH --job-name=ablation_vis
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=23:00:00
#SBATCH --output=/vol/bitbucket/mz325/individual_project/logs/ablation_vis_%j.out
#SBATCH --error=/vol/bitbucket/mz325/individual_project/logs/ablation_vis_%j.err

echo "=========================================="
echo "Comprehensive Ablation Visualization"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "=========================================="

# ── Environment setup ──
eval "$(/vol/bitbucket/mz325/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate /vol/bitbucket/mz325/envs/como

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

cd /vol/bitbucket/mz325/individual_project

mkdir -p vis_results/ablation_visualization
mkdir -p logs

echo ""
echo "--- Running All Visualizations ---"
echo "  Part A: Conv1 filters & feature maps"
echo "  Part B: Original RGB images"
echo "  Part C: Convergence basin comparison grids"
echo "  Part D: 1D cross-section overlays"
echo "  Part E: Channel type classification"
echo ""

python visualize_ablation_results.py

echo ""
echo "=========================================="
echo "Done at $(date)"
echo "Output: vis_results/ablation_visualization/"
echo "=========================================="