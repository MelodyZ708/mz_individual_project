#!/bin/bash
#SBATCH --job-name=como_unet_ate
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=/vol/bitbucket/mz325/individual_project/logs/unet_ate_%j.out
#SBATCH --error=/vol/bitbucket/mz325/individual_project/logs/unet_ate_%j.err

echo "=========================================="
echo " U-Net ATE Experiments - Groups 3/4/5 Only"
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

# 只跑后 3 个配置（前 2 个已完成）
declare -a EXPERIMENTS=(
    "1:d4,d15,d9,d10,d30:enc1_top5"
    "0:d15:enc0_single_Ch15"
    "0:d15,d10,d0:enc0_top3"
)

run_group() {
    local ENC_LEVEL="$1"
    local CHANNELS="$2"
    local EXP_NAME="$3"

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
with open('config/como.yml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
print('Config updated: tracking=unet, enc_level=${ENC_LEVEL}, channels=${CHANNELS}')
"

    local VALID_RUNS=0
    local RMSE_SUM=0

    for RUN in $(seq 1 5); do
        local traj_saved="${RESULTS_DIR}/${EXP_NAME}_run${RUN}.txt"

        # 跳过已有有效结果的 run
        # 有效条件：文件存在 + 行数>=10 + md5与同group所有之前run都不同
        local skip=0
        if [ -f "${traj_saved}" ] && [ "$(wc -l < "${traj_saved}")" -ge 10 ]; then
            local md5_curr
            md5_curr=$(md5sum "${traj_saved}" | awk '{print $1}')
            local is_dup=0
            local prev
            for prev in $(seq 1 $((RUN - 1))); do
                local prev_f="${RESULTS_DIR}/${EXP_NAME}_run${prev}.txt"
                if [ -f "${prev_f}" ]; then
                    local md5_prev
                    md5_prev=$(md5sum "${prev_f}" | awk '{print $1}')
                    if [ "${md5_curr}" = "${md5_prev}" ]; then
                        is_dup=1
                        break
                    fi
                fi
            done
            if [ "${is_dup}" -eq 0 ]; then
                skip=1
            fi
        fi

        if [ "${skip}" -eq 1 ]; then
            local LINES
            LINES=$(wc -l < "${traj_saved}")
            local ATE_VAL
            ATE_VAL=$(grep "rmse" "${RESULTS_DIR}/${EXP_NAME}_run${RUN}_ate.log" 2>/dev/null | awk '{print $2}')
            echo "  --- Run ${RUN}/5 for ${EXP_NAME}: SKIPPED (valid, ${LINES} lines, ATE=${ATE_VAL}) ---"
            VALID_RUNS=$((VALID_RUNS + 1))
            RMSE_SUM=$(python3 -c "print(${RMSE_SUM} + ${ATE_VAL:-0})" 2>/dev/null || echo "${RMSE_SUM}")
            continue
        fi

        echo "  --- Run ${RUN}/5 for ${EXP_NAME} at $(date) ---"

        # --- Headless Display（照抄旧脚本方式，加锁文件清理）---
        local DISP_NUM=$((300 + RUN))
        # 清理可能残留的锁文件
        pkill -f "Xvfb :${DISP_NUM}" 2>/dev/null || true
        sleep 0.3
        rm -f /tmp/.X${DISP_NUM}-lock /tmp/.X11-unix/X${DISP_NUM}
        Xvfb :${DISP_NUM} -screen 0 1920x1080x24 &
        local XVFB_PID=$!
        sleep 1
        export DISPLAY=:${DISP_NUM}

        # 删除旧轨迹文件，防止 Python 失败时复制旧结果
        rm -f "results/tum_rgbd_dataset_freiburg1_desk.txt"

        # Run COMO with timeout
        timeout 300 python como/como_dataset.py \
            --dataset_type=tum \
            --dataset_dir="${DATASET_DIR}" || true

        kill ${XVFB_PID} 2>/dev/null
        wait ${XVFB_PID} 2>/dev/null
        rm -f /tmp/.X${DISP_NUM}-lock /tmp/.X11-unix/X${DISP_NUM}
        sleep 1

        # Save trajectory
        local local_traj="results/tum_rgbd_dataset_freiburg1_desk.txt"

        if [ -f "${local_traj}" ]; then
            cp "${local_traj}" "${traj_saved}"

            # Evaluate ATE
            evo_ape tum "${GT}" "${traj_saved}" \
                --align --correct_scale \
                --save_results "${RESULTS_DIR}/${EXP_NAME}_run${RUN}.zip" \
                > "${RESULTS_DIR}/${EXP_NAME}_run${RUN}_ate.log" 2>&1

            local ATE_VAL
            ATE_VAL=$(grep "rmse" "${RESULTS_DIR}/${EXP_NAME}_run${RUN}_ate.log" | awk '{print $2}')
            echo "      -> Trajectory saved. ATE RMSE: ${ATE_VAL}"

            if [ -n "${ATE_VAL}" ]; then
                VALID_RUNS=$((VALID_RUNS + 1))
                RMSE_SUM=$(python3 -c "print(${RMSE_SUM} + ${ATE_VAL})")
            fi
        else
            echo "      [ERROR] No trajectory file found at ${local_traj}"
        fi
    done

    echo ""
    echo "  ── ${EXP_NAME} SUMMARY (${VALID_RUNS}/5 valid) ──"
    if [ "${VALID_RUNS}" -gt 0 ]; then
        python3 -c "
rmse_sum = ${RMSE_SUM}
n = ${VALID_RUNS}
print(f'  ATE RMSE avg: {rmse_sum/n*100:.3f} cm  (std: N/A for deterministic runs)')
"
    fi
}

for EXP in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r ENC_LEVEL CHANNELS EXP_NAME <<< "$EXP"
    run_group "${ENC_LEVEL}" "${CHANNELS}" "${EXP_NAME}"
done

# --- Restore config ---
cp config/como.yml.bak_unet_ate config/como.yml
echo "Config restored."

# --- Summary (all 5 groups) ---
echo ""
echo "=========================================="
echo " Experiment Summary (ATE RMSE)"
echo "=========================================="
declare -a ALL_EXPERIMENTS=(
    "1:d4:enc1_single_Ch04"
    "1:d4,d15,d9:enc1_top3"
    "1:d4,d15,d9,d10,d30:enc1_top5"
    "0:d15:enc0_single_Ch15"
    "0:d15,d10,d0:enc0_top3"
)
for EXP in "${ALL_EXPERIMENTS[@]}"; do
    IFS=':' read -r ENC_LEVEL CHANNELS EXP_NAME <<< "$EXP"
    echo "Group: ${EXP_NAME}"
    for RUN in 1 2 3 4 5; do
        LOG="${RESULTS_DIR}/${EXP_NAME}_run${RUN}_ate.log"
        TRAJ="${RESULTS_DIR}/${EXP_NAME}_run${RUN}.txt"
        if [ -f "${LOG}" ]; then
            ATE_VAL=$(grep "rmse" "${LOG}" | awk '{print $2}')
            LINES=$(wc -l < "${TRAJ}" 2>/dev/null || echo "?")
            echo "  Run ${RUN}: ATE=${ATE_VAL} (${LINES} lines)"
        else
            echo "  Run ${RUN}: [no result]"
        fi
    done
done

echo ""
echo "All experiments complete at $(date)"