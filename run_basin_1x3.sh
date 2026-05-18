#!/bin/bash
#SBATCH --job-name=basin_1x3
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/vol/bitbucket/mz325/individual_project/logs/basin_1x3_%j.out
#SBATCH --error=/vol/bitbucket/mz325/individual_project/logs/basin_1x3_%j.err

echo "=========================================="
echo "Convergence Basin — 1x3 Per-Combination"
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

mkdir -p vis_results/basin_1x3
mkdir -p logs

echo ""
echo "--- Running Basin 1x3 Visualization ---"
echo "  Frame: 306 (Middle)"
echo "  Conditions: Clean / +30% / +50%"
echo "  Combinations (9):"
echo "    1. Gray (1ch)"
echo "    2. Rank01 Full 8ch"
echo "    3. Rank02 Full 8ch"
echo "    4. [Ch15] (Forward Best *)"
echo "    5. [Ch23] (Forward Best *)"
echo "    6. [Ch6]"
echo "    7. [Ch6, Ch15]"
echo "    8. [Ch8]"
echo "    9. [Ch23, Ch42]"
echo ""

python visualize_basin_1x3.py

echo ""
echo "=========================================="
echo "Done at $(date)"
echo "Output: vis_results/basin_1x3/"
echo "=========================================="