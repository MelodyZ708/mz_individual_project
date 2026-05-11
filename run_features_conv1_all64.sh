#!/bin/bash
#SBATCH --job-name=vis_conv1_64ch
#SBATCH --output=vis_conv1_64ch_%j.out
#SBATCH --error=vis_conv1_64ch_%j.err
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00

echo "=========================================="
echo "Job: Visualize ALL 64 conv1 channels"
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
mkdir -p vis_results/feature_maps_conv1_all64

# Run visualization script
python visualize_features_conv1_all64.py

echo ""
echo "=========================================="
echo "Job completed: $(date)"
echo "Output: vis_results/feature_maps_conv1_all64/"
echo "=========================================="