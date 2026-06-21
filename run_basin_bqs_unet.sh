#!/bin/bash
#SBATCH --job-name=basin_bqs_unet
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/vol/bitbucket/mz325/individual_project/logs/basin_bqs_unet_%j.out
#SBATCH --error=/vol/bitbucket/mz325/individual_project/logs/basin_bqs_unet_%j.err

echo "=========================================="
echo "U-Net Convergence Basin + BQS Evaluation"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "=========================================="

# ── Environment setup ──
eval "$(/vol/bitbucket/mz325/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate /vol/bitbucket/mz325/envs/como

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

# ── Must cd into como/ so that `import como.*` works ──
cd /vol/bitbucket/mz325/individual_project/como

mkdir -p ../vis_results/convergence_basin_unet_enc0
mkdir -p ../vis_results/convergence_basin_unet_enc1
mkdir -p ../vis_results/bqs_unet_enc0
mkdir -p ../vis_results/bqs_unet_enc1
mkdir -p ../logs

# ============================================================
# Step 1: Convergence Basin Visualization — enc0 (16ch, H×W)
# ============================================================
echo ""
echo "--- [1/4] Convergence Basin: U-Net enc0 (16ch) ---"
python /vol/bitbucket/mz325/individual_project/visualize_basin_unet.py \
    --enc_level 0 \
    --frame 306 \
    --rgb_dir /vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/ \
    --model_path models/scannet.ckpt \
    --device cuda:0
echo "enc0 basin done at $(date)"

# ============================================================
# Step 2: Convergence Basin Visualization — enc1 (32ch, H/2)
# ============================================================
echo ""
echo "--- [2/4] Convergence Basin: U-Net enc1 (32ch) ---"
python /vol/bitbucket/mz325/individual_project/visualize_basin_unet.py \
    --enc_level 1 \
    --frame 306 \
    --rgb_dir /vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/ \
    --model_path models/scannet.ckpt \
    --device cuda:0
echo "enc1 basin done at $(date)"

# ============================================================
# Step 3: BQS Evaluation — enc0
# ============================================================
echo ""
echo "--- [3/4] BQS Evaluation: U-Net enc0 ---"
python /vol/bitbucket/mz325/individual_project/compute_bqs_unet.py \
    --enc_level 0 \
    --frame 306 \
    --rgb_dir /vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/ \
    --model_path models/scannet.ckpt \
    --device cuda:0
echo "enc0 BQS done at $(date)"

# ============================================================
# Step 4: BQS Evaluation — enc1
# (Optional: add --resnet_csv if you have the ResNet BQS CSV)
# ============================================================
echo ""
echo "--- [4/4] BQS Evaluation: U-Net enc1 ---"
python /vol/bitbucket/mz325/individual_project/compute_bqs_unet.py \
    --enc_level 1 \
    --frame 306 \
    --rgb_dir /vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/ \
    --model_path models/scannet.ckpt \
    --device cuda:0
echo "enc1 BQS done at $(date)"

# ============================================================
# Summary
# ============================================================
echo ""
echo "=========================================="
echo "All done at $(date)"
echo ""
echo "Outputs:"
echo "  Convergence Basin:"
echo "    ../vis_results/convergence_basin_unet_enc0/channel_00.png ~ channel_15.png"
echo "    ../vis_results/convergence_basin_unet_enc1/channel_00.png ~ channel_31.png"
echo "  BQS Scores:"
echo "    ../vis_results/bqs_unet_enc0/bqs_scores.csv"
echo "    ../vis_results/bqs_unet_enc0/bqs_ranking.png"
echo "    ../vis_results/bqs_unet_enc1/bqs_scores.csv"
echo "    ../vis_results/bqs_unet_enc1/bqs_ranking.png"
echo "=========================================="