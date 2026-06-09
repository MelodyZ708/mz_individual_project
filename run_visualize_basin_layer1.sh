#!/bin/bash
#SBATCH --job-name=vis_layer1
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/vol/bitbucket/mz325/individual_project/logs/vis_layer1_%j.out
#SBATCH --error=/vol/bitbucket/mz325/individual_project/logs/vis_layer1_%j.err

echo "=========================================="
echo "Convergence Basin — All 64 Layer1 Channels"
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

mkdir -p vis_results/convergence_basin_layer1
mkdir -p logs

echo ""
echo "--- Running Layer1 64-Channel Convergence Basin Visualization ---"
echo "  Extractor: conv1 -> bn1 -> relu -> maxpool -> layer1"
echo "  Frame: 306"
echo "  Conditions: Clean, +30%, +50%"
echo "  Channels: 0-63"
echo "  Grid: 61x61 per channel per condition"
echo "  Total: 192 cost landscapes"
echo ""

python visualize_basin_layer1_64ch.py

echo ""
echo "=========================================="
echo "Done at $(date)"
echo "Output: vis_results/convergence_basin_layer1/"
echo "  - channel_00.png ~ channel_63.png"
echo "  - bqs_summary.csv"
echo "  - channel_ranking.csv"
echo "=========================================="