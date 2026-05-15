#!/bin/bash
#SBATCH --job-name=ch_search_5ch
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/vol/bitbucket/mz325/individual_project/logs/channel_search_5ch_%j.out
#SBATCH --error=/vol/bitbucket/mz325/individual_project/logs/channel_search_5ch_%j.err

echo "=========================================="
echo "5-Channel Search — Convergence Basin"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "=========================================="

# ── Environment setup ──
eval "$(/vol/bitbucket/mz325/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate /vol/bitbucket/mz325/envs/como

export MPLBACKEND=Agg

cd /vol/bitbucket/mz325/individual_project

mkdir -p vis_results/channel_search_5ch
mkdir -p logs

echo ""
echo "--- Running 5-Channel Search ---"
echo "  4 fixed (Top1/Top2 de-dead) + 15 random x 5-ch"
echo ""

python random_channel_search_5ch.py --n_random_runs 15

echo ""
echo "=========================================="
echo "Done at $(date)"
echo "Output: vis_results/channel_search_5ch/"
echo "=========================================="