#!/bin/bash
#SBATCH --job-name=como_unet_ate
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=/vol/bitbucket/mz325/individual_project/logs/unet_ate_%j.out
#SBATCH --error=/vol/bitbucket/mz325/individual_project/logs/unet_ate_%j.err

echo "=========================================="
echo " U-Net ATE Experiments (5 Groups x 5 Runs)"
echo " Node: $(hostname)"
echo " GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo " Date: $(date)"
echo "=========================================="

# --- Environment Setup ---
eval "$(/vol/bitbucket/mz325/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate /vol/bitbucket/mz325/envs/como

# --- Navigate to project ---
cd /vol/bitbucket/mz325/individual_project/como

# Backup config
cp config/como.yml config/como.yml.bak_unet_ate

DATASET_DIR="/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/"
GT="/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/groundtruth.txt"
RESULTS_DIR="results/unet_ate"
mkdir -p ${RESULTS_DIR}

# Define the 5 experiment groups
# Format: "enc_level:channel_select:exp_name"
declare -a EXPERIMENTS=(
    "1:d4:enc1_single_Ch04"
    "1:d4,d15,d9:enc1_top3"
    "1:d4,d15,d9,d10,d30:enc1_top5"
    "0:d15:enc0_single_Ch15"
    "0:d15,d10,d0:enc0_top3"
)

for EXP in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r ENC_LEVEL CHANNELS EXP_NAME <<< "$EXP"

    echo "=========================================="
    echo " Starting Experiment Group: ${EXP_NAME}"
    echo " enc_level: ${ENC_LEVEL}, channels: ${CHANNELS}"
    echo "=========================================="

    # Update config for this experiment
    python3 -c "
import yaml
with open('config/como.yml', 'r') as f:
    cfg = yaml.safe_load(f)

cfg['tracking']['color'] = 'unet'
cfg['tracking']['unet_enc_level'] = ${ENC_LEVEL}
cfg['tracking']['unet_channel_select'] = '${CHANNELS}'

# mapping color 保持不变，只改 tracking

with open('config/como.yml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
print('Config updated: tracking=unet, enc_level=${ENC_LEVEL}, channels=${CHANNELS}')
"

    # Run 5 times
    for RUN in 1 2 3 4 5; do
        echo "  --- Run ${RUN}/5 for ${EXP_NAME} at $(date) ---"

        # --- Headless Display ---
        Xvfb :$((200 + RUN)) -screen 0 1920x1080x24 &
        XVFB_PID=$!
        sleep 1
        export DISPLAY=:$((200 + RUN))

        # Run COMO with timeout
        timeout 300 python como/como_dataset.py \
            --dataset_type=tum \
            --dataset_dir="${DATASET_DIR}" || true

        kill ${XVFB_PID} 2>/dev/null

        # Save trajectory (COMO always writes to results/, not results/unet_ate/)
        local_traj="results/tum_rgbd_dataset_freiburg1_desk.txt"
        traj_saved="${RESULTS_DIR}/${EXP_NAME}_run${RUN}.txt"

        if [ -f "${local_traj}" ]; then
            cp "${local_traj}" "${traj_saved}"

            # Evaluate ATE
            evo_ape tum "${GT}" "${traj_saved}" \
                --align --correct_scale \
                --save_results "${RESULTS_DIR}/${EXP_NAME}_run${RUN}.zip" \
                > "${RESULTS_DIR}/${EXP_NAME}_run${RUN}_ate.log" 2>&1

            ATE_VAL=$(grep "rmse" "${RESULTS_DIR}/${EXP_NAME}_run${RUN}_ate.log" | awk '{print $2}')
            echo "      -> Trajectory saved. ATE RMSE: ${ATE_VAL}"
        else
            echo "      [ERROR] No trajectory file found at ${local_traj}"
        fi
    done
done

# --- Restore config ---
cp config/como.yml.bak_unet_ate config/como.yml
echo "Config restored."

# --- Summary ---
echo ""
echo "=========================================="
echo " Experiment Summary (ATE RMSE)"
echo "=========================================="
for EXP in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r ENC_LEVEL CHANNELS EXP_NAME <<< "$EXP"
    echo "Group: ${EXP_NAME}"
    for RUN in 1 2 3 4 5; do
        LOG="${RESULTS_DIR}/${EXP_NAME}_run${RUN}_ate.log"
        if [ -f "${LOG}" ]; then
            ATE_VAL=$(grep "rmse" "${LOG}" | awk '{print $2}')
            echo "  Run ${RUN}: ${ATE_VAL}"
        else
            echo "  Run ${RUN}: [no result]"
        fi
    done
done

echo ""
echo "All experiments complete at $(date)"