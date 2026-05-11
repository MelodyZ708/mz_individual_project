#!/bin/bash
#SBATCH --job-name=como_vis_layer1
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/vol/bitbucket/mz325/individual_project/logs/vis_layer1_%j.out
#SBATCH --error=/vol/bitbucket/mz325/individual_project/logs/vis_layer1_%j.err

echo "=========================================="
echo "Layer1 Visualization"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "=========================================="

eval "$(/vol/bitbucket/mz325/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate /vol/bitbucket/mz325/envs/como

Xvfb :202 -screen 0 1920x1080x24 &
XVFB_PID=$!
sleep 1
export DISPLAY=:202

cd /vol/bitbucket/mz325/individual_project

mkdir -p vis_results/conv1/feature_maps
mkdir -p vis_results/layer1/feature_maps
mkdir -p vis_results/comparison

for f in vis_results/cnn_channels_img*.png vis_results/cnn_activation_img*.png; do
    if [ -f "$f" ]; then
        mv "$f" vis_results/conv1/feature_maps/
        echo "Moved: $f"
    fi
done

echo "--- Running Feature Map Visualization ---"
python visualize_features_layer1.py

echo "--- Running Gradient Comparison ---"
python visualize_gradients_layer1.py

kill $XVFB_PID 2>/dev/null
echo "All done at $(date)"
