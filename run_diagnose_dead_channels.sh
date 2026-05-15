#!/bin/bash
#SBATCH --job-name=dead_ch_diag
#SBATCH --output=dead_ch_diag_%j.out
#SBATCH --error=dead_ch_diag_%j.err
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00

echo "=========================================="
echo "Job: Diagnose Dead Channels in Conv1"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "=========================================="

# Activate environment
eval "$(/vol/bitbucket/mz325/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate /vol/bitbucket/mz325/envs/como

# Set up virtual display for matplotlib
export MPLBACKEND=Agg

# Navigate to project directory
cd /vol/bitbucket/mz325/individual_project

# Create output directory
mkdir -p vis_results/dead_channel_diagnosis

# Run diagnosis script
python diagnose_dead_channels.py

echo ""
echo "=========================================="
echo "Job completed: $(date)"
echo "Output: vis_results/dead_channel_diagnosis/"
echo "=========================================="