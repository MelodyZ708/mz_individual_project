#!/bin/bash
# run_ablation_eval.sh — Greedy Channel Selection Ablation Study (Post-ReLU only)
# 对每个 layer 的 Greedy 搜索路径逐步消融，每个配置跑 5 次取平均 ATE/RPE
# Mapping 固定为 gray，Tracking 用 CNN (Post-ReLU)
#
# Conv1  Post-ReLU: seed=Ch06, 7步 → [6,28,34,62,12,54,3]  (most cost-effective: step4)
# Layer1 Post-ReLU: seed=Ch02, 6步 → [2,61,60,32,53,41]    (most cost-effective: step5)
# Layer2 Post-ReLU: seed=Ch120,3步 → [120,66,39]            (global optimum: step3)
# 共 16 组 × 5 runs = 80 次

set -e

# ── 路径配置 ──
PROJECT_DIR="$HOME/code/individual_project/como"
DATASET_DIR="/home/melody/data/tum/rgbd_dataset_freiburg1_desk/"
GT_FILE="${DATASET_DIR}groundtruth.txt"
CONFIG_FILE="${PROJECT_DIR}/config/como.yml"
CONFIG_BACKUP="${PROJECT_DIR}/config/como.yml.bak"
RESULTS_DIR="${PROJECT_DIR}/results/ablation_greedy"
TRAJ_SRC="${PROJECT_DIR}/results/tum_rgbd_dataset_freiburg1_desk.txt"
NUM_RUNS=5

mkdir -p "${RESULTS_DIR}"
cp "${CONFIG_FILE}" "${CONFIG_BACKUP}"
trap "echo 'Restoring config...'; cp '${CONFIG_BACKUP}' '${CONFIG_FILE}'" EXIT

cd "${PROJECT_DIR}"

# ── 汇总文件（全新创建）──
SUMMARY="${RESULTS_DIR}/summary.txt"
echo -e "Experiment\tLayer\tStep\tNum_Ch\tChannels\tATE_mean(cm)\tATE_std(cm)\tRPE_mean(cm)\tRPE_std(cm)\tValidRuns" > "${SUMMARY}"

# ── 辅助函数 ──
run_experiment() {
    local EXP_NAME="$1"
    local LAYER_NAME="$2"
    local CHANNELS="$3"
    local NUM_CH="$4"
    local STEP="$5"
    local FULL_CH_COUNT="$6"

    echo "  [${EXP_NAME}] layer=${LAYER_NAME} step=${STEP} ch=(${CHANNELS})"

    local SNIP="/tmp/snip_ablation.yml"
    cat > "${SNIP}" <<EOF
tracking:
  color: cnn
  cnn_mode: cnn_only
  cnn_layer_name: ${LAYER_NAME}
  cnn_channels: ${NUM_CH}
  cnn_channel_select: "${CHANNELS}"
  cnn_layer_full_channels: ${FULL_CH_COUNT}
mapping:
  color: gray
EOF

    python3 - <<PYEOF
import yaml
with open("${CONFIG_FILE}") as f: cfg = yaml.safe_load(f)
with open("${SNIP}") as f: snip = yaml.safe_load(f)
for s in ("tracking", "mapping"):
    if s in snip: cfg[s].update(snip[s])
with open("${CONFIG_FILE}", "w") as f: yaml.dump(cfg, f, default_flow_style=False)
PYEOF

    local ATE_VALS=()
    local RPE_VALS=()

    for RUN in $(seq 1 ${NUM_RUNS}); do
        export DISPLAY=:1

        timeout 600 python como/como_dataset.py \
            --dataset_type=tum \
            --dataset_dir="${DATASET_DIR}" || true

        sleep 5   # 等待 GPU 显存释放，防止 CUDA busy

        local SAVED="${RESULTS_DIR}/${EXP_NAME}_run${RUN}.txt"
        if [ -f "${TRAJ_SRC}" ]; then
            mv "${TRAJ_SRC}" "${SAVED}"

            local ATE_RMSE=$(evo_ape tum "${GT_FILE}" "${SAVED}" \
                --align --correct_scale 2>/dev/null | grep "rmse" | awk '{print $2}')
            local RPE_RMSE=$(evo_rpe tum "${GT_FILE}" "${SAVED}" \
                --align --correct_scale 2>/dev/null | grep "rmse" | awk '{print $2}')

            if [ -n "${ATE_RMSE}" ]; then
                ATE_VALS+=("${ATE_RMSE}")
                RPE_VALS+=("${RPE_RMSE:-0}")
                printf "     Run %d: ATE=%.4f m | RPE=%s m\n" "${RUN}" "${ATE_RMSE}" "${RPE_RMSE:-N/A}"
            else
                echo "     Run ${RUN}: ATE eval failed"
            fi
        else
            echo "     Run ${RUN}: No trajectory file"
        fi
    done

    local VALID=${#ATE_VALS[@]}
    if [ "${VALID}" -gt 0 ]; then
        python3 - <<PYEOF2
import math
ate_raw = [${ATE_VALS[@]/%/,}]
rpe_raw = [${RPE_VALS[@]/%/,}]
n = len(ate_raw)
ate = [x*100 for x in ate_raw]
rpe = [x*100 for x in rpe_raw]
ate_mean = sum(ate)/n
ate_std  = math.sqrt(sum((x-ate_mean)**2 for x in ate)/n) if n>1 else 0
rpe_mean = sum(rpe)/n
rpe_std  = math.sqrt(sum((x-rpe_mean)**2 for x in rpe)/n) if n>1 else 0
print(f"  -> ATE={ate_mean:.3f}±{ate_std:.3f} cm | RPE={rpe_mean:.3f}±{rpe_std:.3f} cm ({n}/${NUM_RUNS} valid)")
with open("${SUMMARY}", "a") as f:
    f.write(f"${EXP_NAME}\t${LAYER_NAME}\t${STEP}\t${NUM_CH}\t${CHANNELS}\t{ate_mean:.3f}\t{ate_std:.3f}\t{rpe_mean:.3f}\t{rpe_std:.3f}\t{n}\n")
PYEOF2
    else
        echo "  -> No valid runs for ${EXP_NAME}"
    fi
    echo ""
}

# ════════════════════════════════════════════════════════════
# CONV1 POST-RELU — Greedy Beam 1 (seed=Ch06, 7 steps)
# most cost-effective: Step 4 [6,28,34,62]
# ════════════════════════════════════════════════════════════
echo "=================================================="
echo "CONV1 POST-RELU Ablation (Beam1, seed=Ch06, 7 steps)"
echo "=================================================="

run_experiment "conv1_s1" "conv1" "d6"                          1 1 64
run_experiment "conv1_s2" "conv1" "d6,d28"                      2 2 64
run_experiment "conv1_s3" "conv1" "d6,d28,d34"                  3 3 64
run_experiment "conv1_s4" "conv1" "d6,d28,d34,d62"              4 4 64   # most cost-effective
run_experiment "conv1_s5" "conv1" "d6,d28,d34,d62,d12"          5 5 64
run_experiment "conv1_s6" "conv1" "d6,d28,d34,d62,d12,d54"      6 6 64
run_experiment "conv1_s7" "conv1" "d6,d28,d34,d62,d12,d54,d3"   7 7 64

# ════════════════════════════════════════════════════════════
# LAYER1 POST-RELU — Greedy Beam 1 (seed=Ch02, 6 steps)
# most cost-effective: Step 5 [2,61,60,32,53]
# ════════════════════════════════════════════════════════════
echo "=================================================="
echo "LAYER1 POST-RELU Ablation (Beam1, seed=Ch02, 6 steps)"
echo "=================================================="

run_experiment "layer1_s1" "layer1" "d2"                       1 1 64
run_experiment "layer1_s2" "layer1" "d2,d61"                   2 2 64
run_experiment "layer1_s3" "layer1" "d2,d61,d60"               3 3 64
run_experiment "layer1_s4" "layer1" "d2,d61,d60,d32"           4 4 64
run_experiment "layer1_s5" "layer1" "d2,d61,d60,d32,d53"       5 5 64   # most cost-effective
run_experiment "layer1_s6" "layer1" "d2,d61,d60,d32,d53,d41"  6 6 64

# ════════════════════════════════════════════════════════════
# LAYER2 POST-RELU — Greedy Beam 1 (seed=Ch120, 3 steps)
# global optimum: Step 3 [120,66,39]
# ════════════════════════════════════════════════════════════
echo "=================================================="
echo "LAYER2 POST-RELU Ablation (Beam1, seed=Ch120, 3 steps)"
echo "=================================================="

run_experiment "layer2_s1" "layer2" "d120"         1 1 128
run_experiment "layer2_s2" "layer2" "d120,d66"     2 2 128
run_experiment "layer2_s3" "layer2" "d120,d66,d39" 3 3 128   # global optimum

# ════════════════════════════════════════════════════════════
echo "=================================================="
echo "ABLATION COMPLETE — Full Summary:"
echo "=================================================="
column -t -s $'\t' "${SUMMARY}"
echo ""
echo "Full results saved in: ${RESULTS_DIR}/"