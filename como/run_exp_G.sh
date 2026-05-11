#!/bin/bash
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --job-name=como_G
#SBATCH --output=/vol/bitbucket/mz325/individual_project/logs/como_G_%j.out
#SBATCH --error=/vol/bitbucket/mz325/individual_project/logs/como_G_%j.err
#SBATCH --time=3-00:00:00

eval "$(/vol/bitbucket/mz325/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate /vol/bitbucket/mz325/envs/como

Xvfb :201 -screen 0 1920x1080x24 &
sleep 1
export DISPLAY=:201

cd /vol/bitbucket/mz325/individual_project/como
mkdir -p results
mkdir -p /vol/bitbucket/mz325/individual_project/logs

GT=/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/groundtruth.txt

for i in 1 2 3 4 5; do
    echo "========== Run $i =========="
    python como/como_dataset.py \
        --dataset_type=tum \
        --dataset_dir=/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/
    mv results/tum_rgbd_dataset_freiburg1_desk.txt results/fr1desk_G_cnnonly_ch258_run${i}.txt
    echo "--- ATE for Run $i ---"
    evo_ape tum "$GT" results/fr1desk_G_cnnonly_ch258_run${i}.txt --align --correct_scale
    echo ""
done
echo "========== All 5 runs complete =========="
