#!/bin/bash
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --job-name=como_D6
#SBATCH --output=/vol/bitbucket/mz325/individual_project/logs/como_D6_%j.out
#SBATCH --error=/vol/bitbucket/mz325/individual_project/logs/como_D6_%j.err
#SBATCH --time=01:00:00

eval "$(/vol/bitbucket/mz325/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate /vol/bitbucket/mz325/envs/como

Xvfb :201 -screen 0 1920x1080x24 &
sleep 1
export DISPLAY=:201

cd /vol/bitbucket/mz325/individual_project/como
GT=/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/groundtruth.txt

echo "========== Run 6 (replacement for Run 4) =========="
python como/como_dataset.py \
    --dataset_type=tum \
    --dataset_dir=/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/

mv results/tum_rgbd_dataset_freiburg1_desk.txt results/fr1desk_D_cnnonly_active_run6.txt

echo "--- ATE for Run 6 ---"
evo_ape tum "$GT" results/fr1desk_D_cnnonly_active_run6.txt --align --correct_scale
