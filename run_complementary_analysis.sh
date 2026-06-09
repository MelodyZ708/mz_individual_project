#!/bin/bash
#SBATCH --job-name=comp_analysis
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/vol/bitbucket/mz325/individual_project/logs/comp_analysis_%j.out
#SBATCH --error=/vol/bitbucket/mz325/individual_project/logs/comp_analysis_%j.err

echo "=========================================="
echo "Complementary Analysis — Optimal 4-Channel Subset [06, 28, 34, 62]"
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

mkdir -p vis_results/forward_greedy_bqs/analysis

echo ""
echo "--- 1/3  Feature Maps (Clean vs +50%) ---"
python vis_feature_maps.py

echo ""
echo "--- 2/3  Gradient Vector Field ---"
python vis_gradient_field.py

echo ""
echo "--- 3/3  Ablation Basin Analysis ---"
python vis_ablation_basin.py

echo ""
echo "=========================================="
echo "Done at $(date)"
echo "Outputs: vis_results/forward_greedy_bqs/analysis/"
echo "  feature_maps_optimal4_frame306.png"
echo "  gradient_field_optimal4_frame306.png"
echo "  ablation_basin_optimal4_frame306.png"
echo "=========================================="