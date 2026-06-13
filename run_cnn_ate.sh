#!/bin/bash
#SBATCH --job-name=como_cnn_ate
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/vol/bitbucket/mz325/individual_project/logs/cnn_ate_%j.out
#SBATCH --error=/vol/bitbucket/mz325/individual_project/logs/cnn_ate_%j.err

echo "=========================================="
echo "CoMo CNN ATE Evaluation (Affine Fixed)"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "=========================================="

eval "$(/vol/bitbucket/mz325/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate /vol/bitbucket/mz325/envs/como

Xvfb :202 -screen 0 1920x1080x24 &
XVFB_PID=$!
sleep 1
export DISPLAY=:202

cd /vol/bitbucket/mz325/individual_project/como

mkdir -p results

echo "--- Running CoMo CNN (Affine Fixed) ---"
python como/como_dataset.py \
    --dataset_type=tum \
    --dataset_dir=/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/

kill $XVFB_PID 2>/dev/null

echo "--- Computing ATE ---"
if [ -f results/tum_rgbd_dataset_freiburg1_desk.txt ]; then
    cp results/tum_rgbd_dataset_freiburg1_desk.txt \
       results/fr1desk_cnn_affine_fixed.txt
    evo_ape tum \
        /vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/groundtruth.txt \
        results/fr1desk_cnn_affine_fixed.txt \
        --align --correct_scale
fi

echo "All done at $(date)"
