#!/bin/bash
#SBATCH --job-name=ch_search
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/vol/bitbucket/mz325/individual_project/logs/channel_search_%j.out
#SBATCH --error=/vol/bitbucket/mz325/individual_project/logs/channel_search_%j.err

echo "=========================================="
echo "Random Channel Search — Convergence Basin"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "=========================================="

# ── Environment setup ──
eval "$(/vol/bitbucket/mz325/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate /vol/bitbucket/mz325/envs/como

export MPLBACKEND=Agg

cd /vol/bitbucket/mz325/individual_project

mkdir -p vis_results/channel_search
mkdir -p logs

echo ""
echo "--- Running Random Channel Search ---"
echo "  20 runs x 8-ch + 5 runs x 12-ch + 5 runs x 16-ch"
echo ""

python random_channel_search.py \
    --n_runs 20 \
    --n_channels 8 \
    --extra_sizes "12,16"

echo ""
echo "=========================================="
echo "Done at $(date)"
echo "Output: vis_results/channel_search/"
echo "=========================================="