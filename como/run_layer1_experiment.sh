#!/bin/bash
#SBATCH --job-name=como_L1_cnnonly_layer1
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/vol/bitbucket/mz325/individual_project/logs/L1_cnnonly_layer1_%j.out
#SBATCH --error=/vol/bitbucket/mz325/individual_project/logs/L1_cnnonly_layer1_%j.err

# ============================================================
# Config L1: CNN-only 8ch, layer1 features
# 5 sequential runs in one job
# ============================================================

echo "=========================================="
echo "Config L1: CNN-only 8ch (layer1)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "=========================================="

# --- Environment Setup ---
eval "$(/vol/bitbucket/mz325/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate /vol/bitbucket/mz325/envs/como

# --- Headless Display ---
Xvfb :201 -screen 0 1920x1080x24 &
XVFB_PID=$!
sleep 1
export DISPLAY=:201

# --- Navigate to project ---
cd /vol/bitbucket/mz325/individual_project/como

# --- Configure for Layer1 experiment ---
cp config/como.yml config/como.yml.bak_L1

python3 -c "
import yaml
with open('config/como.yml', 'r') as f:
    cfg = yaml.safe_load(f)

cfg['tracking']['color'] = 'cnn'
cfg['tracking']['cnn_channels'] = 8
cfg['tracking']['cnn_mode'] = 'cnn_only'
cfg['tracking']['cnn_channel_select'] = 'all'
cfg['tracking']['cnn_layer'] = 'layer1'

cfg['mapping']['color'] = 'cnn'
cfg['mapping']['cnn_channels'] = 8
cfg['mapping']['cnn_mode'] = 'cnn_only'
cfg['mapping']['cnn_channel_select'] = 'all'
cfg['mapping']['cnn_layer'] = 'layer1'

with open('config/como.yml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)

print('Config updated: cnn_only, 8ch, layer1')
"

GT="/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/groundtruth.txt"
RESULT_DIR="results"
mkdir -p ${RESULT_DIR}

# --- Run 5 times ---
for RUN in 1 2 3 4 5; do
    echo ""
    echo "=========================================="
    echo "Run ${RUN}/5 starting at $(date)"
    echo "=========================================="

    python como/como_dataset.py \
        --dataset_type=tum \
        --dataset_dir=/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/

    # Find and save the latest trajectory
    TRAJ_FILE=$(ls -t results/*.txt 2>/dev/null | head -1)
    if [ -n "$TRAJ_FILE" ]; then
        cp "$TRAJ_FILE" "${RESULT_DIR}/fr1desk_L1_cnnonly_layer1_run${RUN}.txt"
        echo "Trajectory saved: fr1desk_L1_cnnonly_layer1_run${RUN}.txt"

        # Evaluate
        echo "--- ATE Evaluation (Run ${RUN}) ---"
        evo_ape tum "$GT" \
            "${RESULT_DIR}/fr1desk_L1_cnnonly_layer1_run${RUN}.txt" \
            --align --correct_scale
    else
        echo "[ERROR] Run ${RUN}: No trajectory file found!"
    fi
done

# --- Restore config ---
cp config/como.yml.bak_L1 config/como.yml
echo ""
echo "Config restored. All 5 runs complete at $(date)"

# --- Cleanup ---
kill $XVFB_PID 2>/dev/null