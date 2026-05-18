#!/bin/bash
#SBATCH --job-name=debug_act
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=04:30:00
#SBATCH --output=/vol/bitbucket/mz325/individual_project/logs/debug_activations_%j.out
#SBATCH --error=/vol/bitbucket/mz325/individual_project/logs/debug_activations_%j.err

echo "=========================================="
echo "Debug: Channel Activation Statistics"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "=========================================="

eval "$(/vol/bitbucket/mz325/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate /vol/bitbucket/mz325/envs/como

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

cd /vol/bitbucket/mz325/individual_project

mkdir -p logs

python debug_channel_activations.py

echo ""
echo "=========================================="
echo "Done at $(date)"
echo "=========================================="