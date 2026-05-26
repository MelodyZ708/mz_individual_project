#!/bin/bash
#SBATCH --job-name=basin_met
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=23:00:00
#SBATCH --output=/vol/bitbucket/mz325/individual_project/logs/basin_metrics_%j.out
#SBATCH --error=/vol/bitbucket/mz325/individual_project/logs/basin_metrics_%j.err

echo "=========================================="
echo "Convergence Basin — Metric Evaluation"
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

mkdir -p vis_results/basin_metrics/raw_data
mkdir -p vis_results/basin_metrics/plots
mkdir -p logs

echo ""
echo "--- Stage 1: Computing Cost Grids (GPU) ---"
echo "  Dataset: freiburg1_desk"
echo "  Frame: 306"
echo "  Channels: 0-63 (Conv1+BN+ReLU)"
echo "  Conditions: Clean, +30%, +50%"
echo "  Grid: 61x61 (±30px)"
echo "  Output: vis_results/basin_metrics/raw_data/*.npy"
echo ""

python compute_basin_data.py

echo ""
echo "--- Stage 2: Analyzing Metrics (CPU) ---"
echo "  Metrics: R², Basin Width, CSR, Sharpness, Condition Number"
echo "  Output: vis_results/basin_metrics/metrics_summary.csv"
echo "          vis_results/basin_metrics/channel_ranking.csv"
echo "          vis_results/basin_metrics/plots/*.png"
echo ""

python analyze_basin_metrics.py

echo ""
echo "=========================================="
echo "Done at $(date)"
echo "Output: vis_results/basin_metrics/"
echo "=========================================="