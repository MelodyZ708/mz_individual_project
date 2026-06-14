#!/bin/bash
#SBATCH --job-name=vis_bqs
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/vis_bqs_%j.out

echo "=========================================="
echo "Visualize Best Channel Combinations"
echo "Node: $SLURM_NODELIST"
echo "GPU:  $CUDA_VISIBLE_DEVICES"
echo "Date: $(date)"
echo "=========================================="

eval "$(/vol/bitbucket/mz325/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate /vol/bitbucket/mz325/envs/como

cd /vol/bitbucket/mz325/individual_project

# conv1 pre-relu  →  BQS=0.7807  [6, 28, 34, 50, 39, 16, 52, 2]
echo "--- conv1 pre-relu ---"
python visualize_best_combination.py --layer conv1 --relu_mode pre --channels 6 28 34 50 39 16 52 2

# conv1 post-relu  →  BQS=0.7403  [6, 28, 34, 62, 12, 54, 3, 2]
echo "--- conv1 post-relu ---"
python visualize_best_combination.py --layer conv1 --relu_mode post --channels 6 28 34 62 12 54 3 2

# layer1 pre-relu  →  BQS=0.6791  [2, 0]
echo "--- layer1 pre-relu ---"
python visualize_best_combination.py --layer layer1 --relu_mode pre --channels 2 0

# layer1 post-relu  →  BQS=0.6264  [2, 61, 60, 32, 53, 41]
echo "--- layer1 post-relu ---"
python visualize_best_combination.py --layer layer1 --relu_mode post --channels 2 61 60 32 53 41

# layer2 pre-relu  →  BQS=0.6749  [112, 75, 40, 43]
echo "--- layer2 pre-relu ---"
python visualize_best_combination.py --layer layer2 --relu_mode pre --channels 112 75 40 43

# layer2 post-relu  →  BQS=0.6974  [120, 66, 39]
echo "--- layer2 post-relu ---"
python visualize_best_combination.py --layer layer2 --relu_mode post --channels 120 66 39

echo "All visualizations completed."