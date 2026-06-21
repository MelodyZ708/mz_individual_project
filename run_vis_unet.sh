#!/bin/bash
#SBATCH --job-name=como_vis_unet
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/vol/bitbucket/mz325/individual_project/logs/vis_unet_%j.out
#SBATCH --error=/vol/bitbucket/mz325/individual_project/logs/vis_unet_%j.err

echo "=========================================="
echo "U-Net Feature Map Visualization (P3)"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "=========================================="

eval "$(/vol/bitbucket/mz325/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate /vol/bitbucket/mz325/envs/como

Xvfb :203 -screen 0 1920x1080x24 &
XVFB_PID=$!
sleep 1
export DISPLAY=:203

cd /vol/bitbucket/mz325/individual_project/como

mkdir -p vis_results/unet_feature_maps

echo "--- Running U-Net Feature Map Visualization ---"
echo "enc_level=1 (32ch, H/2xW/2), frame=306"
python ../vis_unet_feature_maps.py \
    --rgb_dir /vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/ \
    --ckpt models/scannet.ckpt \
    --frame 306 \
    --enc_level 1 \
    --out_dir ../vis_results/unet_feature_maps \
    --device cuda

echo ""
echo "--- Also running enc_level=0 (16ch, HxW) for comparison ---"
python ../vis_unet_feature_maps.py \
    --rgb_dir /vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/ \
    --ckpt models/scannet.ckpt \
    --frame 306 \
    --enc_level 0 \
    --out_dir ../vis_results/unet_feature_maps \
    --device cuda

kill $XVFB_PID 2>/dev/null
echo "=========================================="
echo "All done at $(date)"
echo "Outputs in: /vol/bitbucket/mz325/individual_project/vis_results/unet_feature_maps/"
echo "=========================================="