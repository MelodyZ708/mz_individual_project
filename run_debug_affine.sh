#!/bin/bash
#SBATCH --job-name=como_debug_affine
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/vol/bitbucket/mz325/individual_project/logs/debug_affine_%j.out
#SBATCH --error=/vol/bitbucket/mz325/individual_project/logs/debug_affine_%j.err

echo "=========================================="
echo "CoMo CNN Debug - Affine Inspection"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "=========================================="

eval "$(/vol/bitbucket/mz325/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate /vol/bitbucket/mz325/envs/como

cd /vol/bitbucket/mz325/individual_project/como

mkdir -p vis_results

echo "--- Running CoMo headless (CNN+CNN mode) ---"
python como_dataset_headless.py \
    --dataset_type=tum \
    --dataset_dir=/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/

echo "--- Copying debug outputs to logs ---"
if [ -f vis_results/mapping_photo_err.txt ]; then
    cp vis_results/mapping_photo_err.txt \
       /vol/bitbucket/mz325/individual_project/logs/mapping_photo_err_${SLURM_JOB_ID}.txt
    echo "Saved: mapping_photo_err_${SLURM_JOB_ID}.txt"
fi

echo "All done at $(date)"
