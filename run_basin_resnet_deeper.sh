#!/bin/bash
#SBATCH --job-name=como_basin_resnet_deeper
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/vol/bitbucket/mz325/individual_project/logs/basin_resnet_deeper_%j.out
#SBATCH --error=/vol/bitbucket/mz325/individual_project/logs/basin_resnet_deeper_%j.err

echo "=========================================="
echo "Convergence Basin — ResNet-18 layer3 & layer4"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "=========================================="

eval "$(/vol/bitbucket/mz325/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate /vol/bitbucket/mz325/envs/como

Xvfb :204 -screen 0 1920x1080x24 &
XVFB_PID=$!
sleep 1
export DISPLAY=:204

cd /vol/bitbucket/mz325/individual_project/como

mkdir -p vis_results/convergence_basin_resnet_layer3
mkdir -p vis_results/convergence_basin_resnet_layer4

# ------------------------------------------------------------------
# layer3: 256 channels, H/16×W/16, upsample 16×
# Runtime estimate: ~256 channels × 3 conditions × 61×61 grid
#                   ≈ 2–3 h on a40 (dominated by cost landscape loop)
# ------------------------------------------------------------------
echo ""
echo "--- [1/2] ResNet-18 layer3 (256ch) ---"
echo "Frame=306, grid=61×61, shift=±30px"
python ../visualize_basin_resnet_deeper.py \
    --layer layer3 \
    --frame 306 \
    --rgb_dir /vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/ \
    --out_dir vis_results/convergence_basin_resnet_layer3 \
    --device cuda

echo ""
echo "--- [2/2] ResNet-18 layer4 (512ch) ---"
echo "Frame=306, grid=61×61, shift=±30px"
python ../visualize_basin_resnet_deeper.py \
    --layer layer4 \
    --frame 306 \
    --rgb_dir /vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/ \
    --out_dir vis_results/convergence_basin_resnet_layer4 \
    --device cuda

kill $XVFB_PID 2>/dev/null

echo ""
echo "=========================================="
echo "All done at $(date)"
echo "Outputs:"
echo "  layer3 → vis_results/convergence_basin_resnet_layer3/"
echo "  layer4 → vis_results/convergence_basin_resnet_layer4/"
echo "=========================================="