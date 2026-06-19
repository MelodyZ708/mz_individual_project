#!/bin/bash
# run_layer2_s3.sh — 单独补跑 layer2_s3 (3通道 [120,66,39])
# 结果追加到已有的 ablation_greedy/summary.txt

set -e

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

SUMMARY="${RESULTS_DIR}/summary.txt"

# ── 写入 layer2_s3 配置 ──
cat > /tmp/snip_layer2_s3.yml <<EOF
tracking:
  color: cnn
  cnn_mode: cnn_only
  cnn_layer_name: layer2
  cnn_channels: 3
  cnn_channel_select: "d120,d66,d39"
  cnn_layer_full_channels: 128
mapping:
  color: gray
EOF

python3 - <<PYEOF
import yaml
with open("${CONFIG_FILE}") as f: cfg = yaml.safe_load(f)
with open("/tmp/snip_layer2_s3.yml") as f: snip = yaml.safe_load(f)
for s in ("tracking", "mapping"):
    if s in snip: cfg[s].update(snip[s])
with open("${CONFIG_FILE}", "w") as f: yaml.dump(cfg, f, default_flow_style=False)
PYEOF

echo "=================================================="
echo "LAYER2 POST-RELU s3 — [120,66,39] (global optimum)"
echo "=================================================="

ATE_VALS=()
RPE_VALS=()

for RUN in $(seq 1 ${NUM_RUNS}); do
    export DISPLAY=:1

    timeout 600 python como/como_dataset.py \
        --dataset_type=tum \
        --dataset_dir="${DATASET_DIR}" || true

    sleep 5

    SAVED="${RESULTS_DIR}/layer2_s3_run${RUN}.txt"
    if [ -f "${TRAJ_SRC}" ]; then
        mv "${TRAJ_SRC}" "${SAVED}"

        ATE_RMSE=$(evo_ape tum "${GT_FILE}" "${SAVED}" \
            --align --correct_scale 2>/dev/null | grep "rmse" | awk '{print $2}')
        RPE_RMSE=$(evo_rpe tum "${GT_FILE}" "${SAVED}" \
            --align --correct_scale 2>/dev/null | grep "rmse" | awk '{print $2}')

        if [ -n "${ATE_RMSE}" ]; then
            ATE_VALS+=("${ATE_RMSE}")
            RPE_VALS+=("${RPE_RMSE:-0}")
            printf "     Run %d: ATE=%.4f m | RPE=%s m\n" "${RUN}" "${ATE_RMSE}" "${RPE_RMSE:-N/A}"
        else
            echo "     Run ${RUN}: ATE eval failed"
        fi
    else
        echo "     Run ${RUN}: No trajectory file (NaN divergence)"
    fi
done

VALID=${#ATE_VALS[@]}
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
    f.write(f"layer2_s3\tlayer2\t3\t3\td120,d66,d39\t{ate_mean:.3f}\t{ate_std:.3f}\t{rpe_mean:.3f}\t{rpe_std:.3f}\t{n}\n")
PYEOF2
else
    echo "  -> All runs failed (NaN divergence in layer2)"
    python3 - <<PYEOF3
with open("${SUMMARY}", "a") as f:
    f.write("layer2_s3\tlayer2\t3\t3\td120,d66,d39\tN/A\tN/A\tN/A\tN/A\t0\n")
PYEOF3
fi

echo ""
echo "Done. Summary:"
column -t -s $'\t' "${SUMMARY}"