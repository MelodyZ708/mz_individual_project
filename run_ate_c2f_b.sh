#!/bin/bash
#SBATCH --job-name=como_c2f_b
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --output=/vol/bitbucket/mz325/individual_project/logs/ate_c2f_b_%j.out
#SBATCH --error=/vol/bitbucket/mz325/individual_project/logs/ate_c2f_b_%j.err

echo "=========================================="
echo "ATE Evaluation: C2F-B (L0/L1 coarse, L2/L3 fine)"
echo "Datasets: flashlight + lightswitch"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=========================================="

eval "$(/vol/bitbucket/mz325/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate /vol/bitbucket/mz325/envs/como

PROJECT_DIR="/vol/bitbucket/mz325/individual_project/como"
CONFIG_FILE="${PROJECT_DIR}/config/como.yml"
CONFIG_BACKUP="${PROJECT_DIR}/config/como.yml.bak"
RESULTS_DIR="${PROJECT_DIR}/results/ate_c2f_comparison"
NUM_RUNS=5

mkdir -p "${RESULTS_DIR}"
mkdir -p "/vol/bitbucket/mz325/individual_project/logs"

# ── 备份原始 config，脚本结束时恢复 ──
cp "${CONFIG_FILE}" "${CONFIG_BACKUP}"
trap "echo '[Cleanup] Restoring original config...'; cp '${CONFIG_BACKUP}' '${CONFIG_FILE}'; echo '[Cleanup] Done.'" EXIT

cd "${PROJECT_DIR}"

# ══════════════════════════════════════════════════════════════
# 辅助函数：对单个数据集运行 N 次，计算平均 ATE + RPE
# 用法: run_experiment <实验名> <数据集路径> <config_snippet_file>
# ══════════════════════════════════════════════════════════════
run_experiment() {
    local EXP_NAME="$1"
    local DATASET_DIR="$2"
    local CONFIG_SNIPPET="$3"

    local GT_FILE="${DATASET_DIR}/groundtruth.txt"
    local DATASET_BASENAME
    DATASET_BASENAME=$(basename "${DATASET_DIR}")
    # COMO 固定输出到 results/datasets_tum.txt
    local TRAJ_FILE="${PROJECT_DIR}/results/datasets_tum.txt"

    echo ""
    echo "##################################################"
    echo "# EXPERIMENT : ${EXP_NAME}"
    echo "# DATASET    : ${DATASET_BASENAME}"
    echo "##################################################"

    # ── 把 snippet 合并进 como.yml（只覆盖 tracking 端）──
    python3 - <<PYEOF
import yaml

with open("${CONFIG_FILE}", "r") as f:
    cfg = yaml.safe_load(f)

with open("${CONFIG_SNIPPET}", "r") as f:
    snippet = yaml.safe_load(f)

if "tracking" in snippet:
    cfg["tracking"].update(snippet["tracking"])

with open("${CONFIG_FILE}", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

print(f"[Config] Applied snippet for ${EXP_NAME}")
print(f"  tracking.color            = {cfg['tracking'].get('color')}")
print(f"  tracking.cnn_layer_coarse = {cfg['tracking'].get('cnn_layer_coarse', 'N/A')}")
print(f"  tracking.cnn_layer_fine   = {cfg['tracking'].get('cnn_layer_fine', 'N/A')}")
print(f"  tracking.cnn_c2f_version = {cfg['tracking'].get('cnn_c2f_version', 'NOT SET!')}")
assert str(cfg['tracking'].get('cnn_c2f_version', '')).upper() == 'B', '[ERROR] Expected C2F-B'
PYEOF

    local RMSE_SUM=0
    local MEAN_SUM=0
    local RPE_RMSE_SUM=0
    local VALID_RUNS=0

    for RUN in $(seq 1 ${NUM_RUNS}); do
        echo ""
        echo "  ── ${EXP_NAME} | Run ${RUN}/${NUM_RUNS} ──"

        # 删除上次残留的轨迹文件，防止 fail 时误读旧结果
        rm -f "${TRAJ_FILE}"

        local DISP_NUM=$((300 + RUN))
        Xvfb :${DISP_NUM} -screen 0 1920x1080x24 &
        local XVFB_PID=$!
        sleep 1
        export DISPLAY=:${DISP_NUM}

        timeout 300 python como/como_dataset.py \
            --dataset_type=tum \
            --dataset_dir="${DATASET_DIR}" \
            || true

        kill ${XVFB_PID} 2>/dev/null

        local TRAJ_SAVED="${RESULTS_DIR}/${EXP_NAME}_run${RUN}.txt"

        if [ -f "${TRAJ_FILE}" ]; then
            cp "${TRAJ_FILE}" "${TRAJ_SAVED}"

            local ATE_OUT
            ATE_OUT=$(evo_ape tum \
                "${GT_FILE}" "${TRAJ_SAVED}" \
                --align --correct_scale 2>&1)

            local RMSE MEAN
            RMSE=$(echo "${ATE_OUT}" | grep -E "^\s*rmse" | awk '{print $2}')
            MEAN=$(echo "${ATE_OUT}" | grep -E "^\s*mean" | awk '{print $2}')

            local RPE_OUT
            RPE_OUT=$(evo_rpe tum \
                "${GT_FILE}" "${TRAJ_SAVED}" \
                --align --correct_scale 2>&1)

            local RPE_RMSE
            RPE_RMSE=$(echo "${RPE_OUT}" | grep -E "^\s*rmse" | awk '{print $2}')

            if [ -n "${RMSE}" ]; then
                RMSE_SUM=$(python3 -c "print(${RMSE_SUM} + ${RMSE})")
                MEAN_SUM=$(python3 -c "print(${MEAN_SUM} + ${MEAN})")
                RPE_RMSE_SUM=$(python3 -c "print(${RPE_RMSE_SUM} + ${RPE_RMSE:-0})")
                VALID_RUNS=$((VALID_RUNS + 1))
                echo "  Run ${RUN}: ATE_RMSE=${RMSE} m  |  RPE_RMSE=${RPE_RMSE:-N/A} m"
            else
                echo "  Run ${RUN}: ATE parse failed — check trajectory"
                echo "${ATE_OUT}" | head -20
            fi
        else
            echo "  Run ${RUN}: Trajectory file not found (COMO likely failed/timed out)"
        fi
    done

    echo ""
    echo "  ── ${EXP_NAME} | ${DATASET_BASENAME} | SUMMARY (${VALID_RUNS}/${NUM_RUNS} valid) ──"
    if [ "${VALID_RUNS}" -gt 0 ]; then
        python3 - <<PYEOF2
rmse_sum = ${RMSE_SUM}
mean_sum = ${MEAN_SUM}
rpe_sum  = ${RPE_RMSE_SUM}
n        = ${VALID_RUNS}
exp      = "${EXP_NAME}"
ds       = "${DATASET_BASENAME}"
print(f"  ATE RMSE (avg over {n} runs): {rmse_sum/n*100:.3f} cm")
print(f"  ATE Mean (avg over {n} runs): {mean_sum/n*100:.3f} cm")
print(f"  RPE RMSE (avg over {n} runs): {rpe_sum/n*100:.3f} cm")
summary_file = "${RESULTS_DIR}/summary.tsv"
with open(summary_file, "a") as f:
    f.write(f"{ds}\t{exp}\t{rmse_sum/n*100:.3f}\t{mean_sum/n*100:.3f}\t{rpe_sum/n*100:.3f}\t{n}\n")
PYEOF2
    else
        echo "  No valid runs — skipping summary entry."
    fi
}


# ══════════════════════════════════════════════════════════════
# C2F-B snippet：L0/L1 用 coarse（layer2），L2/L3 用 fine（conv1）
# 对应 Tracking.py 中改动后的一行：
#   cnn_c2f_version: B selects L0/L1 coarse and L2/L3 fine.
# ══════════════════════════════════════════════════════════════
cat > /tmp/snip_c2f_b.yml <<'EOF'
tracking:
  color: cnn_c2f
  cnn_c2f_version: B
  cnn_layer_coarse: layer2
  cnn_channels_coarse: 3
  cnn_channel_select_coarse: "d120,d66,d54"
  cnn_layer_fine: conv1
  cnn_channels_fine: 6
  cnn_channel_select_fine: "d6,d28,d34,d50,d39,d16"
  cnn_layer_full_channels: 64
  cnn_mode: cnn_only
EOF

DS_FLASHLIGHT="/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk_flashlight"
DS_LIGHTSWITCH="/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk_lightswitch"

echo ""
echo "=========================================="
echo "DATASET 1/2: flashlight"
echo "=========================================="
run_experiment "c2f_b_L01coarse_L23fine" "${DS_FLASHLIGHT}" /tmp/snip_c2f_b.yml

echo ""
echo "=========================================="
echo "DATASET 2/2: lightswitch"
echo "=========================================="
run_experiment "c2f_b_L01coarse_L23fine" "${DS_LIGHTSWITCH}" /tmp/snip_c2f_b.yml


# ══════════════════════════════════════════════════════════════
# 最终汇总打印
# ══════════════════════════════════════════════════════════════
echo ""
echo "=========================================="
echo "FINAL SUMMARY  (C2F-B only)"
echo "=========================================="
echo ""
python3 - <<'PYEOF'
import os
from collections import defaultdict

summary_file = "/vol/bitbucket/mz325/individual_project/como/results/ate_c2f_comparison/summary.tsv"
if not os.path.exists(summary_file):
    print("[ERROR] summary.tsv not found")
    exit(1)

with open(summary_file) as f:
    lines = f.readlines()

rows = [l.strip().split('\t') for l in lines[1:] if l.strip()]

# 只打印本次新增的 c2f_b 行
target = "c2f_b_L01coarse_L23fine"
groups = defaultdict(list)
for row in rows:
    if len(row) == 6 and row[1] == target:
        groups[row[0]].append(row)

if not groups:
    print("  No results found for c2f_b_L01coarse_L23fine — check if runs completed.")
else:
    col_w = [42, 15, 15, 15, 10]
    sep   = "-+-".join(["-" * w for w in col_w])
    for ds, ds_rows in groups.items():
        print(f"\n  Dataset: {ds}")
        print(f"  {'Experiment':<42} {'ATE RMSE(cm)':>15} {'ATE Mean(cm)':>15} {'RPE RMSE(cm)':>15} {'Valid':>10}")
        print(f"  {sep}")
        for row in ds_rows:
            _, exp, ate_rmse, ate_mean, rpe_rmse, n = row
            print(f"  {exp:<42} {ate_rmse:>15} {ate_mean:>15} {rpe_rmse:>15} {n:>10}")

print(f"\nFull results appended to: {summary_file}")
PYEOF

echo ""
echo "Individual trajectories: ${RESULTS_DIR}/c2f_b_L01coarse_L23fine_run<N>.txt"
echo "All done at $(date)"
echo "=========================================="