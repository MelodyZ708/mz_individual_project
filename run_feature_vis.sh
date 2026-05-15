#!/bin/bash
#SBATCH --job-name=feature_vis
#SBATCH --output=feature_vis_%j.out
#SBATCH --error=feature_vis_%j.err
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G

# Initialize micromamba (use full path)
export MAMBA_ROOT_PREFIX="/vol/bitbucket/mz325/micromamba"
eval "$(/vol/bitbucket/mz325/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate /vol/bitbucket/mz325/envs/como

# Set paths
PROJECT_DIR="/vol/bitbucket/mz325/individual_project"
DATASET_DIR="/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk"
IMAGE_PATH="${DATASET_DIR}/rgb/1305031452.891726.png"
OUTPUT_DIR="${PROJECT_DIR}/feature_vis_output"

cd ${PROJECT_DIR}

python visualize_top_channels.py \
    --image_path "$IMAGE_PATH" \
    --output_dir "$OUTPUT_DIR"

echo "Done! Output in: $OUTPUT_DIR"