#!/bin/bash
#SBATCH --job-name=c2f_c_single
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/vol/bitbucket/mz325/individual_project/logs/c2f_c_single_%j.out
#SBATCH --error=/vol/bitbucket/mz325/individual_project/logs/c2f_c_single_%j.err

echo "=========================================="
echo "ATE Evaluation: C2F-C (L0 coarse, L1/L2/L3 fine) — single run each"
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
TRAJ_FILE="${PROJECT_DIR}/results/datasets_tum.txt"

mkdir -p "${RESULTS_DIR}"
mkdir -p "/vol/bitbucket/mz325/individual_project/logs"

# ── 备份原始 config，脚本结束时恢复 ──
cp "${CONFIG_FILE}" "${CONFIG_BACKUP}"
trap "echo '[Cleanup] Restoring original config...'; cp '${CONFIG_BACKUP}' '${CONFIG_FILE}'; echo '[Cleanup] Done.'" EXIT

cd "${PROJECT_DIR}"

# ── 写 C2F-C snippet（cnn_c2f_version=C：只有 L0 用 coarse，L1/L2/L3 全用 fine）──
cat > /tmp/snip_c2f_c.yml <<'EOF'
tracking:
  color: cnn_c2f
  cnn_c2f_version: C
  cnn_layer_coarse: layer2
  cnn_channels_coarse: 3
  cnn_channel_select_coarse: "d120,d66,d54"
  cnn_layer_fine: conv1
  cnn_channels_fine: 6
  cnn_channel_select_fine: "d6,d28,d34,d50,d39,d16"
  cnn_layer_full_channels: 64
  cnn_mode: cnn_only
EOF

# ── 合并 snippet 进 como.yml ──
python3 - <<'PYEOF'
import yaml

config_file = "/vol/bitbucket/mz325/individual_project/como/config/como.yml"

with open(config_file, "r") as f:
    cfg = yaml.safe_load(f)

with open("/tmp/snip_c2f_c.yml", "r") as f:
    snippet = yaml.safe_load(f)

if "tracking" in snippet:
    cfg["tracking"].update(snippet["tracking"])

with open(config_file, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

print("[Config] Applied C2F-C snippet")
print(f"  tracking.color               = {cfg['tracking'].get('color')}")
print(f"  tracking.cnn_layer_coarse    = {cfg['tracking'].get('cnn_layer_coarse', 'N/A')}")
print(f"  tracking.cnn_layer_fine      = {cfg['tracking'].get('cnn_layer_fine', 'N/A')}")
print(f"  tracking.cnn_c2f_version = {cfg['tracking'].get('cnn_c2f_version', 'NOT SET!')}")
assert str(cfg['tracking'].get('cnn_c2f_version', '')).upper() == 'C', \
    f"[ERROR] Expected C2F-C, got {cfg['tracking'].get('cnn_c2f_version')}"
PYEOF

# ══════════════════════════════════════════════════════════════
# 辅助函数：单次运行并计算 ATE/RPE
# ══════════════════════════════════════════════════════════════
run_single() {
    local LABEL="$1"
    local DATASET_DIR="$2"
    local SAVE_NAME="$3"
    local GT_FILE="${DATASET_DIR}/groundtruth.txt"

    echo ""
    echo "=========================================="
    echo "Running C2F-C on: $(basename ${DATASET_DIR})"
    echo "=========================================="

    # 删除残留轨迹，防止 fail 时误读
    rm -f "${TRAJ_FILE}"

    local DISP_NUM=$((300 + RANDOM % 100))
    Xvfb :${DISP_NUM} -screen 0 1920x1080x24 &
    local XVFB_PID=$!
    sleep 1
    export DISPLAY=:${DISP_NUM}

    timeout 900 python como/como_dataset.py \
        --dataset_type=tum \
        --dataset_dir="${DATASET_DIR}" \
        || true

    kill ${XVFB_PID} 2>/dev/null

    echo ""
    if [ -f "${TRAJ_FILE}" ]; then
        local TRAJ_SAVED="${RESULTS_DIR}/${SAVE_NAME}.txt"
        cp "${TRAJ_FILE}" "${TRAJ_SAVED}"
        echo "Trajectory saved to: ${TRAJ_SAVED}"

        echo ""
        echo "── ATE (${LABEL}) ──"
        evo_ape tum "${GT_FILE}" "${TRAJ_SAVED}" --align --correct_scale

        echo ""
        echo "── RPE (${LABEL}) ──"
        evo_rpe tum "${GT_FILE}" "${TRAJ_SAVED}" --align --correct_scale
    else
        echo "[FAIL] ${LABEL}: Trajectory file not found — COMO likely crashed or timed out."
    fi
}

# ── 分别跑两个数据集 ──
run_single "flashlight" \
    "/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk_flashlight" \
    "c2f_c_flashlight_single"

run_single "lightswitch" \
    "/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk_lightswitch" \
    "c2f_c_lightswitch_single"

echo ""
echo "All done at $(date)"
echo "=========================================="