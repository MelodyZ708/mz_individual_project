#!/bin/bash
#SBATCH --job-name=basin_64ch
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=23:00:00
#SBATCH --output=/vol/bitbucket/mz325/individual_project/logs/basin_64ch_%j.out
#SBATCH --error=/vol/bitbucket/mz325/individual_project/logs/basin_64ch_%j.err

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

mkdir -p vis_results/convergence_basin_64ch
mkdir -p logs

echo ""
echo "--- Running 64-Channel Convergence Basin Visualization ---"
echo "  Frame: 306 (middle)"
echo "  Conditions: Clean, +30%, +50%"
echo "  Channels: 0-63 (layer1 output)"
echo "  Grid: 61x61 per channel per condition"
echo "  Total: 192 cost landscapes"
echo ""

python visualize_basin_all64_channels.py

echo ""
echo "=========================================="
echo "Done at $(date)"
echo "Output: vis_results/convergence_basin_64ch/"
echo "  - channel_00.png ~ channel_63.png"
echo "  - sharpness_summary.csv"
echo "=========================================="