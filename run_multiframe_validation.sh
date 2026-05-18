#!/bin/bash
#SBATCH --job-name=mf_validate
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/vol/bitbucket/mz325/individual_project/logs/multiframe_validate_%j.out
#SBATCH --error=/vol/bitbucket/mz325/individual_project/logs/multiframe_validate_%j.err

echo "=========================================="
echo "Multi-Frame Channel Validation"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "=========================================="

# ── Environment setup ──
eval "$(/vol/bitbucket/mz325/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate /vol/bitbucket/mz325/envs/como

export MPLBACKEND=Agg

cd /vol/bitbucket/mz325/individual_project

mkdir -p vis_results/multiframe_validation
mkdir -p logs

echo ""
echo "--- Running Multi-Frame Validation ---"
echo "  20 frames x 10 combinations (Top 10 from initial search)"
echo "  Including grayscale baseline"
echo ""

python validate_channels_multiframe.py \
    --n_frames 20

echo ""
echo "=========================================="
echo "Done at $(date)"
echo "Output: vis_results/multiframe_validation/"
echo "=========================================="