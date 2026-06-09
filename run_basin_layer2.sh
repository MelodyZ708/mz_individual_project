#!/bin/bash
#SBATCH --job-name=basin_layer2
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=/vol/bitbucket/mz325/individual_project/logs/basin_layer2_%j.out
#SBATCH --error=/vol/bitbucket/mz325/individual_project/logs/basin_layer2_%j.err

echo "=========================================="
echo "Stage 3: Layer2 Convergence Basin Analysis"
echo "  128 channels, 3D surface visualisation"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "=========================================="

# ── Activate environment ──
eval "$(/vol/bitbucket/mz325/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate /vol/bitbucket/mz325/envs/como

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /vol/bitbucket/mz325/individual_project

# ── Create output directories ──
mkdir -p vis_results/convergence_basin_layer2
mkdir -p logs

echo ""
echo "--- Step 1: Extract Layer2 features + compute BQS + generate 3D basin plots ---"
echo "  Input:  conv1->bn1->relu->maxpool->layer1->layer2  (128 channels)"
echo "  Frame:  306"
echo "  Grid:   61x61, shift +/-30px"
echo "  Output: vis_results/convergence_basin_layer2/"
echo ""

python visualize_basin_layer2_128ch.py

echo ""
echo "--- Step 2: Generate Top-10 summary plots and ranking table ---"
echo ""

python visualize_layer2_topN.py \
    --topn 10 \
    --outdir vis_results/convergence_basin_layer2

echo ""
echo "=========================================="
echo "Done at $(date)"
echo ""
echo "Output files:"
echo "  vis_results/convergence_basin_layer2/channel_000.png ... channel_127.png"
echo "  vis_results/convergence_basin_layer2/bqs_summary.csv"
echo "  vis_results/convergence_basin_layer2/channel_ranking.csv"
echo "  vis_results/convergence_basin_layer2/layer2_bqs_all_channels.png"
echo "  vis_results/convergence_basin_layer2/layer2_component_breakdown.png"
echo "  vis_results/convergence_basin_layer2/layer2_bqs_vs_kill.png"
echo "=========================================="