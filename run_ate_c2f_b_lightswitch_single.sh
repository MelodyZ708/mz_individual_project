#!/bin/bash
#SBATCH --job-name=c2f_b_ls
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/vol/bitbucket/mz325/individual_project/logs/c2f_b_ls_%j.out
#SBATCH --error=/vol/bitbucket/mz325/individual_project/logs/c2f_b_ls_%j.err

echo "=========================================="
echo "ATE Evaluation: C2F-B on lightswitch (single run)"
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
DATASET_DIR="/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk_lightswitch"
GT_FILE="${DATASET_DIR}/groundtruth.txt"
TRAJ_FILE="${PROJECT_DIR}/results/datasets_tum.txt"

mkdir -p "${RESULTS_DIR}"
mkdir -p "/vol/bitbucket/mz325/individual_project/logs"

# ── 备份原始 config，脚本结束时恢复 ──
cp "${CONFIG_FILE}" "${CONFIG_BACKUP}"
trap "echo '[Cleanup] Restoring original config...'; cp '${CONFIG_BACKUP}' '${CONFIG_FILE}'; echo '[Cleanup] Done.'" EXIT

cd "${PROJECT_DIR}"

# ── 写 C2F-B snippet ──
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

# ── 合并 snippet 进 como.yml ──
python3 - <<'PYEOF'
import yaml

config_file = "/vol/bitbucket/mz325/individual_project/como/config/como.yml"

with open(config_file, "r") as f:
    cfg = yaml.safe_load(f)

with open("/tmp/snip_c2f_b.yml", "r") as f:
    snippet = yaml.safe_load(f)

if "tracking" in snippet:
    cfg["tracking"].update(snippet["tracking"])

with open(config_file, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

print("[Config] Applied C2F-B snippet")
print(f"  tracking.color               = {cfg['tracking'].get('color')}")
print(f"  tracking.cnn_layer_coarse    = {cfg['tracking'].get('cnn_layer_coarse', 'N/A')}")
print(f"  tracking.cnn_layer_fine      = {cfg['tracking'].get('cnn_layer_fine', 'N/A')}")
print(f"  tracking.cnn_c2f_version = {cfg['tracking'].get('cnn_c2f_version', 'NOT SET!')}")
assert str(cfg['tracking'].get('cnn_c2f_version', '')).upper() == 'B', \
    f"[ERROR] Expected C2F-B, got {cfg['tracking'].get('cnn_c2f_version')}"
PYEOF

# ── 删除残留轨迹，防止 fail 时误读 ──
rm -f "${TRAJ_FILE}"

# ── 启动虚拟显示 ──
Xvfb :301 -screen 0 1920x1080x24 &
XVFB_PID=$!
sleep 1
export DISPLAY=:301

echo ""
echo "── Running C2F-B on lightswitch ──"
timeout 300 python como/como_dataset.py \
    --dataset_type=tum \
    --dataset_dir="${DATASET_DIR}" \
    || true

kill ${XVFB_PID} 2>/dev/null

# ── 计算 ATE / RPE ──
echo ""
if [ -f "${TRAJ_FILE}" ]; then
    TRAJ_SAVED="${RESULTS_DIR}/c2f_b_lightswitch_single.txt"
    cp "${TRAJ_FILE}" "${TRAJ_SAVED}"
    echo "Trajectory saved to: ${TRAJ_SAVED}"

    echo ""
    echo "── ATE ──"
    evo_ape tum "${GT_FILE}" "${TRAJ_SAVED}" --align --correct_scale

    echo ""
    echo "── RPE ──"
    evo_rpe tum "${GT_FILE}" "${TRAJ_SAVED}" --align --correct_scale
else
    echo "[FAIL] Trajectory file not found — COMO likely crashed or timed out."
fi

echo ""
echo "All done at $(date)"
echo "=========================================="