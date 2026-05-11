#!/bin/bash
#SBATCH --job-name=como_basin_trans
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/vol/bitbucket/mz325/individual_project/logs/basin_translation_%j.out
#SBATCH --error=/vol/bitbucket/mz325/individual_project/logs/basin_translation_%j.err

echo "=========================================="
echo "Convergence Basin — Translation (X/Y Pixel Shift)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "=========================================="

eval "$(/vol/bitbucket/mz325/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate /vol/bitbucket/mz325/envs/como

Xvfb :204 -screen 0 1920x1080x24 &
XVFB_PID=$!
sleep 1
export DISPLAY=:204

cd /vol/bitbucket/mz325/individual_project

# --- Ensure config uses conv1 (not layer1) ---
CONFIG_FILE="como/config/como.yml"
echo "--- Checking cnn_layer in config ---"
grep "cnn_layer" "$CONFIG_FILE" || echo "  (cnn_layer not found in config)"

# Backup config, force conv1
cp "$CONFIG_FILE" "${CONFIG_FILE}.bak_basin_trans"
sed -i 's/cnn_layer: layer1/cnn_layer: conv1/' "$CONFIG_FILE"
echo "--- Config after fix ---"
grep "cnn_layer" "$CONFIG_FILE"

mkdir -p vis_results/convergence_basin_0.5_translation
mkdir -p logs

echo ""
echo "--- Running Translation convergence basin ---"
echo "  Grid: 61x61, Range: ±30 px"
echo "  Conditions: Clean, Brightness +30%, Noise σ=0.05"
echo "  Frames: 3 (early/middle/late)"
echo ""
python visualize_convergence_basin_translation.py

# Restore config
cp "${CONFIG_FILE}.bak_basin_trans" "$CONFIG_FILE"
echo "--- Config restored ---"

kill $XVFB_PID 2>/dev/null

echo ""
echo "=========================================="
echo "Done at $(date)"
echo "Output: vis_results/convergence_basin_translation/"
echo "=========================================="