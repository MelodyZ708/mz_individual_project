#!/bin/bash
#SBATCH --job-name=como_c2f_comparison
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=16:00:00
#SBATCH --output=/vol/bitbucket/mz325/individual_project/logs/c2f_comparison_%j.out
#SBATCH --error=/vol/bitbucket/mz325/individual_project/logs/c2f_comparison_%j.err

echo "=========================================="
echo "CoMo C2F Comparison: C2F-A / Gray / Conv1 / Layer2"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=========================================="

eval "$(/vol/bitbucket/mz325/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate /vol/bitbucket/mz325/envs/como

PROJECT_DIR="/vol/bitbucket/mz325/individual_project/como"
DATASET_DIR="/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/"
GT_FILE="${DATASET_DIR}/groundtruth.txt"
CONFIG_FILE="${PROJECT_DIR}/config/como.yml"
CONFIG_BACKUP="${PROJECT_DIR}/config/como.yml.bak"
RESULTS_DIR="${PROJECT_DIR}/results"
NUM_RUNS=5

mkdir -p "${RESULTS_DIR}"
mkdir -p "$(dirname ${SBATCH_OUTPUT:-/vol/bitbucket/mz325/individual_project/logs/placeholder})"

# ── 备份原始 config，脚本结束时恢复 ──
cp "${CONFIG_FILE}" "${CONFIG_BACKUP}"
trap "echo 'Restoring original config...'; cp '${CONFIG_BACKUP}' '${CONFIG_FILE}'" EXIT

cd "${PROJECT_DIR}"

# ══════════════════════════════════════════════════════════════
# 辅助函数：运行 N 次并计算平均 ATE + RPE
# 用法: run_experiment <实验名> <config_snippet_file>
# ══════════════════════════════════════════════════════════════
run_experiment() {
    local EXP_NAME="$1"
    local CONFIG_SNIPPET="$2"   # 一个临时 yml 文件，只含 tracking: 和 mapping: 的覆盖项

    echo ""
    echo "##################################################"
    echo "# EXPERIMENT: ${EXP_NAME}"
    echo "##################################################"

    # 用 Python 把 snippet 合并进 como.yml（覆盖 tracking/mapping 的 color 相关 key）
    python3 - <<PYEOF
import yaml, copy

with open("${CONFIG_FILE}", "r") as f:
    cfg = yaml.safe_load(f)

with open("${CONFIG_SNIPPET}", "r") as f:
    snippet = yaml.safe_load(f)

# 深度合并 snippet 到 cfg
for section in ("tracking", "mapping"):
    if section in snippet:
        cfg[section].update(snippet[section])

with open("${CONFIG_FILE}", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

print(f"[Config] Applied snippet for ${EXP_NAME}")
print(f"  tracking.color = {cfg['tracking'].get('color')}")
print(f"  mapping.color  = {cfg['mapping'].get('color')}")
PYEOF

    local RMSE_SUM=0
    local MEAN_SUM=0
    local RPE_RMSE_SUM=0
    local VALID_RUNS=0

    for RUN in $(seq 1 ${NUM_RUNS}); do
        echo ""
        echo "  ── ${EXP_NAME} | Run ${RUN}/${NUM_RUNS} ──"

        # 启动虚拟显示（COMO 需要，即使 headless 也保险起见）
        local DISP_NUM=$((300 + RUN))
        Xvfb :${DISP_NUM} -screen 0 1920x1080x24 &
        local XVFB_PID=$!
        sleep 1
        export DISPLAY=:${DISP_NUM}

        # 运行 COMO
        timeout 900 python como/como_dataset.py \
            --dataset_type=tum \
            --dataset_dir="${DATASET_DIR}" || true

        kill ${XVFB_PID} 2>/dev/null

        # 保存轨迹
        local TRAJ="${RESULTS_DIR}/tum_rgbd_dataset_freiburg1_desk.txt"
        local TRAJ_SAVED="${RESULTS_DIR}/${EXP_NAME}_run${RUN}.txt"

        if [ -f "${TRAJ}" ]; then
            cp "${TRAJ}" "${TRAJ_SAVED}"

            # ── ATE ──
            local ATE_RESULT
            ATE_RESULT=$(evo_ape tum \
                "${GT_FILE}" "${TRAJ_SAVED}" \
                --align --correct_scale 2>&1)

            local RMSE MEAN
            RMSE=$(echo "${ATE_RESULT}" | grep "rmse" | awk '{print $2}')
            MEAN=$(echo "${ATE_RESULT}" | grep "mean" | awk '{print $2}')

            # ── RPE（帧间，衡量 Jittering）──
            local RPE_RESULT
            RPE_RESULT=$(evo_rpe tum \
                "${GT_FILE}" "${TRAJ_SAVED}" \
                --align --correct_scale 2>&1)

            local RPE_RMSE
            RPE_RMSE=$(echo "${RPE_RESULT}" | grep "rmse" | awk '{print $2}')

            if [ -n "${RMSE}" ]; then
                RMSE_SUM=$(python3 -c "print(${RMSE_SUM} + ${RMSE})")
                MEAN_SUM=$(python3 -c "print(${MEAN_SUM} + ${MEAN})")
                RPE_RMSE_SUM=$(python3 -c "print(${RPE_RMSE_SUM} + ${RPE_RMSE:-0})")
                VALID_RUNS=$((VALID_RUNS + 1))
                echo "  Run ${RUN}: ATE_RMSE=${RMSE} m | RPE_RMSE=${RPE_RMSE:-N/A} m"
            else
                echo "  Run ${RUN}: ATE failed (degenerate trajectory?)"
            fi
        else
            echo "  Run ${RUN}: Trajectory file not found"
        fi
    done

    # ── 汇总 ──
    echo ""
    echo "  ── ${EXP_NAME} SUMMARY (${VALID_RUNS}/${NUM_RUNS} valid) ──"
    if [ "${VALID_RUNS}" -gt 0 ]; then
        python3 - <<PYEOF2
rmse_sum    = ${RMSE_SUM}
mean_sum    = ${MEAN_SUM}
rpe_sum     = ${RPE_RMSE_SUM}
n           = ${VALID_RUNS}
exp_name    = "${EXP_NAME}"
print(f"  {exp_name}")
print(f"    ATE RMSE (avg): {rmse_sum/n*100:.3f} cm")
print(f"    ATE Mean (avg): {mean_sum/n*100:.3f} cm")
print(f"    RPE RMSE (avg): {rpe_sum/n*100:.3f} cm")
# 追加到汇总文件
with open("${RESULTS_DIR}/summary_c2f_comparison.txt", "a") as f:
    f.write(f"{exp_name}\t{rmse_sum/n*100:.3f}\t{mean_sum/n*100:.3f}\t{rpe_sum/n*100:.3f}\t{n}\n")
PYEOF2
    fi
}


# ══════════════════════════════════════════════════════════════
# 初始化汇总文件
# ══════════════════════════════════════════════════════════════
echo -e "Experiment\tATE_RMSE_cm\tATE_Mean_cm\tRPE_RMSE_cm\tValid_Runs" \
    > "${RESULTS_DIR}/summary_c2f_comparison.txt"


# ══════════════════════════════════════════════════════════════
# 实验 1：C2F-A（最高优先级，先跑验证）
#   Tracking: cnn_c2f（Level 0/1 用 Layer2 粗定位，Level 2 用 Conv1 精对齐）
#   Mapping:  cnn（用 Conv1，保持 Mapping 稳定）
# ══════════════════════════════════════════════════════════════
cat > /tmp/snippet_c2f_a.yml <<'EOF'
tracking:
  color: cnn_c2f
  cnn_c2f_version: A
  cnn_layer_coarse: "layer2"
  cnn_channels_coarse: 3
  cnn_channel_select_coarse: "d120,d66,d39"
  cnn_layer_fine: "conv1"
  cnn_channels_fine: 6
  cnn_channel_select_fine: "d6,d28,d34,d50,d39,d16"
mapping:
  color: cnn
  cnn_mode: cnn_only
  cnn_layer_name: conv1
  cnn_channels: 6
  cnn_channel_select: "d6,d28,d34,d50,d39,d16"
  cnn_layer_full_channels: 64
EOF
run_experiment "C2F_A_Layer2coarse_Conv1fine" /tmp/snippet_c2f_a.yml


# ══════════════════════════════════════════════════════════════
# 实验 2：Gray Baseline
# ══════════════════════════════════════════════════════════════
cat > /tmp/snippet_gray.yml <<'EOF'
tracking:
  color: gray
mapping:
  color: gray
EOF
run_experiment "Gray_Baseline" /tmp/snippet_gray.yml


# ══════════════════════════════════════════════════════════════
# 实验 3：Conv1-only（Pre-ReLU，6通道最优组合）
# ══════════════════════════════════════════════════════════════
cat > /tmp/snippet_conv1.yml <<'EOF'
tracking:
  color: cnn
  cnn_mode: cnn_only
  cnn_layer_name: conv1
  cnn_channels: 6
  cnn_channel_select: "d6,d28,d34,d50,d39,d16"
  cnn_layer_full_channels: 64
mapping:
  color: cnn
  cnn_mode: cnn_only
  cnn_layer_name: conv1
  cnn_channels: 6
  cnn_channel_select: "d6,d28,d34,d50,d39,d16"
  cnn_layer_full_channels: 64
EOF
run_experiment "Conv1_only_6ch" /tmp/snippet_conv1.yml


# ══════════════════════════════════════════════════════════════
# 实验 4：Layer2-only（Post-ReLU，3通道最优组合）
# ══════════════════════════════════════════════════════════════
cat > /tmp/snippet_layer2.yml <<'EOF'
tracking:
  color: cnn
  cnn_mode: cnn_only
  cnn_layer_name: layer2
  cnn_channels: 3
  cnn_channel_select: "d120,d66,d39"
  cnn_layer_full_channels: 128
mapping:
  color: cnn
  cnn_mode: cnn_only
  cnn_layer_name: layer2
  cnn_channels: 3
  cnn_channel_select: "d120,d66,d39"
  cnn_layer_full_channels: 128
EOF
run_experiment "Layer2_only_3ch" /tmp/snippet_layer2.yml


# ══════════════════════════════════════════════════════════════
# 最终汇总打印
# ══════════════════════════════════════════════════════════════
echo ""
echo "=========================================="
echo "FINAL SUMMARY"
echo "=========================================="
echo ""
echo "Experiment                      | ATE RMSE (cm) | ATE Mean (cm) | RPE RMSE (cm) | Valid"
echo "--------------------------------|---------------|---------------|---------------|------"
python3 - <<'PYEOF'
with open("/vol/bitbucket/mz325/individual_project/como/results/summary_c2f_comparison.txt") as f:
    lines = f.readlines()
for line in lines[1:]:   # skip header
    parts = line.strip().split('\t')
    if len(parts) == 5:
        name, ate_rmse, ate_mean, rpe_rmse, n = parts
        print(f"{name:<32}| {ate_rmse:>13} | {ate_mean:>13} | {rpe_rmse:>13} | {n}")
PYEOF

echo ""
echo "Full results saved to: ${RESULTS_DIR}/summary_c2f_comparison.txt"
echo "Individual trajectories saved as: ${RESULTS_DIR}/<EXP_NAME>_run<N>.txt"
echo ""
echo "All done at $(date)"