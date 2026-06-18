#!/bin/bash
# run_local_eval.sh — 本地 Single-Layer CNN Tracking 验证
# 使用 como_dataset.py（带 GUI），需要有显示器或 Xvfb

set -e

# ── 路径配置（按你本地实际情况修改）──
PROJECT_DIR="$HOME/code/individual_project/como"
DATASET_DIR="/home/melody/data/tum/rgbd_dataset_freiburg1_desk/"
GT_FILE="${DATASET_DIR}/groundtruth.txt"
CONFIG_FILE="${PROJECT_DIR}/config/como.yml"
CONFIG_BACKUP="${PROJECT_DIR}/config/como.yml.bak"
RESULTS_DIR="${PROJECT_DIR}/results/eval_single_layer"
TRAJ_SRC="${PROJECT_DIR}/results/tum_rgbd_dataset_freiburg1_desk.txt"
NUM_RUNS=5

mkdir -p "${RESULTS_DIR}"
cp "${CONFIG_FILE}" "${CONFIG_BACKUP}"
trap "echo 'Restoring config...'; cp '${CONFIG_BACKUP}' '${CONFIG_FILE}'" EXIT

cd "${PROJECT_DIR}"

# ── 辅助函数 ──
run_experiment() {
    local EXP_NAME="$1"
    local SNIPPET="$2"

    echo "=================================================="
    echo "Experiment: ${EXP_NAME}"
    echo "=================================================="

    # 合并 snippet 到 como.yml
    python3 - <<PYEOF
import yaml
with open("${CONFIG_FILE}") as f: cfg = yaml.safe_load(f)
with open("${SNIPPET}") as f: snip = yaml.safe_load(f)
for s in ("tracking", "mapping"):
    if s in snip: cfg[s].update(snip[s])
with open("${CONFIG_FILE}", "w") as f: yaml.dump(cfg, f, default_flow_style=False)
print(f"[Config] tracking.color={cfg['tracking']['color']}, mapping.color={cfg['mapping']['color']}")
PYEOF

    local RMSE_SUM=0; local RPE_SUM=0; local VALID=0

    for RUN in $(seq 1 ${NUM_RUNS}); do
        echo "  -> Run ${RUN}/${NUM_RUNS}..."

        # 如果本地有显示器，直接用 DISPLAY=:0；没有则用 Xvfb
        # 有显示器时注释掉下面两行 Xvfb 相关代码
        export DISPLAY=:1


        timeout 600 python como/como_dataset.py \
            --dataset_type=tum \
            --dataset_dir="${DATASET_DIR}" || true

        # 有 Xvfb 时杀掉
        # kill ${XVFB_PID} 2>/dev/null || true

        local SAVED="${RESULTS_DIR}/${EXP_NAME}_run${RUN}.txt"
        if [ -f "${TRAJ_SRC}" ]; then
            mv "${TRAJ_SRC}" "${SAVED}"

            local ATE_RMSE=$(evo_ape tum "${GT_FILE}" "${SAVED}" \
                --align --correct_scale 2>/dev/null | grep "rmse" | awk '{print $2}')
            local RPE_RMSE=$(evo_rpe tum "${GT_FILE}" "${SAVED}" \
                --align --correct_scale 2>/dev/null | grep "rmse" | awk '{print $2}')

            if [ -n "${ATE_RMSE}" ]; then
                RMSE_SUM=$(python3 -c "print(${RMSE_SUM}+${ATE_RMSE})")
                RPE_SUM=$(python3 -c "print(${RPE_SUM}+${RPE_RMSE:-0})")
                VALID=$((VALID+1))
                printf "     ATE RMSE: %.4f m | RPE RMSE: %s m\n" "${ATE_RMSE}" "${RPE_RMSE:-N/A}"
            else
                echo "     ATE failed"
            fi
        else
            echo "     No trajectory file found"
        fi
    done

    [ "${VALID}" -gt 0 ] && python3 - <<PYEOF2
n=${VALID}
print(f"  [${EXP_NAME}] ATE={${RMSE_SUM}/n*100:.3f} cm | RPE={${RPE_SUM}/n*100:.3f} cm ({n}/${NUM_RUNS} valid)")
with open("${RESULTS_DIR}/summary.txt","a") as f:
    f.write(f"${EXP_NAME}\t{${RMSE_SUM}/n*100:.3f}\t{${RPE_SUM}/n*100:.3f}\t{n}\n")
PYEOF2
    echo ""
}

# ── 初始化汇总 ──
echo -e "Experiment\tATE_RMSE(cm)\tRPE_RMSE(cm)\tValidRuns" > "${RESULTS_DIR}/summary.txt"

# ── 实验 1: Gray Baseline ──
cat > /tmp/snip_gray.yml <<'EOF'
tracking:
  color: gray
mapping:
  color: gray
EOF
run_experiment "Gray_Baseline" /tmp/snip_gray.yml

# ── 实验 2: Conv1, Greedy 最优 4 通道 ──
cat > /tmp/snip_conv1.yml <<'EOF'
tracking:
  color: cnn
  cnn_mode: cnn_only
  cnn_layer_name: conv1
  cnn_channels: 4
  cnn_channel_select: "d6,d28,d34,d62"
mapping:
  color: gray
EOF
run_experiment "Conv1_4ch" /tmp/snip_conv1.yml

# ── 实验 3: Layer1, Greedy 最优 5 通道 ──
cat > /tmp/snip_layer1.yml <<'EOF'
tracking:
  color: cnn
  cnn_mode: cnn_only
  cnn_layer_name: layer1
  cnn_channels: 5
  cnn_channel_select: "d2,d61,d60,d32,d53"
mapping:
  color: gray
EOF
run_experiment "Layer1_5ch" /tmp/snip_layer1.yml

# ── 实验 4: Layer2, Greedy 最优 3 通道 ──
cat > /tmp/snip_layer2.yml <<'EOF'
tracking:
  color: cnn
  cnn_mode: cnn_only
  cnn_layer_name: layer2
  cnn_channels: 3
  cnn_channel_select: "d120,d66,d39"
mapping:
  color: gray
EOF
run_experiment "Layer2_3ch" /tmp/snip_layer2.yml

echo "=================================================="
echo "SUMMARY:"
column -t -s $'\t' "${RESULTS_DIR}/summary.txt"
echo "Trajectories saved in: ${RESULTS_DIR}/"
