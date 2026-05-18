#!/bin/bash
#SBATCH --job-name=fwd_vis
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=23:00:00
#SBATCH --output=/vol/bitbucket/mz325/individual_project/logs/forward_visualization_%j.out
#SBATCH --error=/vol/bitbucket/mz325/individual_project/logs/forward_visualization_%j.err

echo "=========================================="
echo "Forward Greedy Selection -- Visualization"
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

mkdir -p vis_results/forward_visualization
mkdir -p logs

echo ""
echo "--- Running Forward Visualization ---"
echo "  Parts: A (Filters+Features), B (RGB), C (3D Basins),"
echo "         E (Classification), F (Type Analysis)"
echo "  Combinations: Gray, [15], [23], [6,15], [15,23],"
echo "                Rank01-8ch, Rank02-8ch"
echo "  Frames: 41, 306, 512"
echo ""

python visualize_forward_results.py

echo ""
echo "=========================================="
echo "Done at $(date)"
echo "Output: vis_results/forward_visualization/"
echo "=========================================="