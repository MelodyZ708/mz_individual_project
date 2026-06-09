#!/bin/bash
#SBATCH --job-name=vis_fmaps_s2s3
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/vol/bitbucket/mz325/individual_project/logs/vis_fmaps_s2s3_%j.out
#SBATCH --error=/vol/bitbucket/mz325/individual_project/logs/vis_fmaps_s2s3_%j.err

echo "=========================================="
echo "Feature Map Visualisation — Stage 2 & 3"
echo "  Stage 2: Layer1 Top-10 channels"
echo "  Stage 3: Layer2 Top-10 channels"
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

mkdir -p vis_results/feature_maps_layer1
mkdir -p vis_results/feature_maps_layer2
mkdir -p logs

echo ""
echo "--- Step 1: Stage 2 (Layer1) Top-10 Feature Maps ---"
echo "  Channels: [60, 2, 55, 61, 53, 41, 7, 8, 46, 47]"
echo "  Conditions: Clean / +30% / +50%"
echo "  Output: vis_results/feature_maps_layer1/"
echo ""

python vis_feature_maps_layer1.py

echo ""
echo "--- Step 2: Stage 3 (Layer2) Top-10 Feature Maps ---"
echo "  Channels: [39, 66, 120, 58, 81, 18, 106, 43, 123, 40]"
echo "  Conditions: Clean / +30% / +50%"
echo "  Output: vis_results/feature_maps_layer2/"
echo ""

python vis_feature_maps_layer2.py

echo ""
echo "=========================================="
echo "Done at $(date)"
echo ""
echo "Output files:"
echo "  vis_results/feature_maps_layer1/"
echo "    feature_maps_layer1_top10_frame306.png      (3 rows x 10 cols)"
echo "    feature_maps_layer1_top10_diff_frame306.png (diff maps)"
echo "    feature_maps_layer1_top10_stats_frame306.png (statistics)"
echo "  vis_results/feature_maps_layer2/"
echo "    feature_maps_layer2_top10_frame306.png      (3 rows x 10 cols)"
echo "    feature_maps_layer2_top10_diff_frame306.png (diff maps)"
echo "    feature_maps_layer2_top10_stats_frame306.png (statistics)"
echo "    stage2_vs_stage3_bqs_top10.png              (cross-stage BQS comparison)"
echo "=========================================="