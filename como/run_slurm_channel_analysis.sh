#!/bin/bash
#SBATCH --job-name=como_channel_analysis
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=20:00:00
#SBATCH --output=/vol/bitbucket/mz325/individual_project/logs/channel_analysis_%j.out
#SBATCH --error=/vol/bitbucket/mz325/individual_project/logs/channel_analysis_%j.err

# ══════════════════════════════════════════════════════════════
# 实验设计说明：
#
# 从消融实验结果中识别出三类关键节点，针对每个节点设计验证实验：
#
# 【类型A】加入某通道后 ATE 显著改善 → 单独评估该通道 + 与当前最优 subset 组合
# 【类型B】加入某通道后 ATE 显著恶化 → 剔除该通道后重新评估 subset
# 【类型C】当前各 layer 的全局最优 subset（作为参考 baseline）
#
# ── Conv1 跳变节点 ──
#   s2→s3: 加 Ch34 后 ATE 7.96→9.52 (恶化 +1.56cm) [类型B]
#   s4→s5: 加 Ch12 后 ATE 7.92→8.18 (轻微恶化)
#   s5→s6: 加 Ch54 后 ATE 8.18→6.78 (改善 -1.40cm) [类型A]
#   s6→s7: 加 Ch03 后 ATE 6.78→8.74 (恶化 +1.96cm) [类型B]
#
# ── Layer1 跳变节点 ──
#   s1→s2: 加 Ch61 后 ATE 15.88→7.32 (改善 -8.56cm) [类型A]
#   s3→s4: 加 Ch32 后 ATE 7.11→7.25 (轻微恶化)
#   s4→s5: 加 Ch53 后 ATE 7.25→6.76 (改善 -0.49cm) [类型A]
#   s5→s6: 加 Ch41 后 ATE 6.76→5.91 (改善 -0.85cm) [类型A]
#
# ── Layer2 跳变节点 ──
#   s1→s2: 加 Ch66 后 ATE 31.65→6.23 (改善 -25.42cm) [类型A]
#   s2→s3: 加 Ch39 后 ATE 6.23→15.91 (恶化 +9.68cm) [类型B]
# ══════════════════════════════════════════════════════════════

echo "=========================================="
echo "CoMo Channel Analysis: Jump-Point Investigation"
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
RESULTS_DIR="${PROJECT_DIR}/results/eval_channel_analysis"
NUM_RUNS=5

mkdir -p "${RESULTS_DIR}"
mkdir -p "$(dirname ${SBATCH_OUTPUT:-/vol/bitbucket/mz325/individual_project/logs/placeholder})"

cp "${CONFIG_FILE}" "${CONFIG_BACKUP}"
trap "echo 'Restoring original config...'; cp '${CONFIG_BACKUP}' '${CONFIG_FILE}'" EXIT

cd "${PROJECT_DIR}"

run_experiment() {
    local EXP_NAME="$1"
    local CONFIG_SNIPPET="$2"

    echo ""
    echo "##################################################"
    echo "# EXPERIMENT: ${EXP_NAME}"
    echo "##################################################"

    python3 - <<PYEOF
import yaml
with open("${CONFIG_FILE}", "r") as f: cfg = yaml.safe_load(f)
with open("${CONFIG_SNIPPET}", "r") as f: snippet = yaml.safe_load(f)
for section in ("tracking", "mapping"):
    if section in snippet: cfg[section].update(snippet[section])
with open("${CONFIG_FILE}", "w") as f: yaml.dump(cfg, f, default_flow_style=False)
PYEOF

    local RMSE_SUM=0; local RPE_SUM=0; local VALID_RUNS=0

    for RUN in $(seq 1 ${NUM_RUNS}); do
        echo "  ── Run ${RUN}/${NUM_RUNS} ──"
        local DISP_NUM=$((300 + RUN))
        Xvfb :${DISP_NUM} -screen 0 1920x1080x24 &
        local XVFB_PID=$!; sleep 1; export DISPLAY=:${DISP_NUM}

        timeout 300 python como/como_dataset.py \
            --dataset_type=tum \
            --dataset_dir="${DATASET_DIR}" || true

        kill ${XVFB_PID} 2>/dev/null || true
        sleep 5

        local TRAJ="${PROJECT_DIR}/results/tum_rgbd_dataset_freiburg1_desk.txt"
        local TRAJ_SAVED="${RESULTS_DIR}/${EXP_NAME}_run${RUN}.txt"

        if [ -f "${TRAJ}" ]; then
            mv "${TRAJ}" "${TRAJ_SAVED}"
            local RMSE=$(evo_ape tum "${GT_FILE}" "${TRAJ_SAVED}" --align --correct_scale 2>/dev/null | grep "rmse" | awk '{print $2}')
            local RPE=$(evo_rpe tum "${GT_FILE}" "${TRAJ_SAVED}" --align --correct_scale 2>/dev/null | grep "rmse" | awk '{print $2}')
            if [ -n "${RMSE}" ]; then
                RMSE_SUM=$(python3 -c "print(${RMSE_SUM} + ${RMSE})")
                RPE_SUM=$(python3 -c "print(${RPE_SUM} + ${RPE:-0})")
                VALID_RUNS=$((VALID_RUNS + 1))
                echo "  ATE=${RMSE} m | RPE=${RPE:-N/A} m"
            else
                echo "  ATE failed"
            fi
        else
            echo "  No trajectory (NaN divergence?)"
        fi
    done

    if [ "${VALID_RUNS}" -gt 0 ]; then
        python3 - <<PYEOF2
n=${VALID_RUNS}; rs=${RMSE_SUM}; rp=${RPE_SUM}
print(f"  -> ATE={rs/n*100:.3f} cm | RPE={rp/n*100:.3f} cm ({n}/${NUM_RUNS} valid)")
with open("${RESULTS_DIR}/summary.txt", "a") as f:
    f.write(f"${EXP_NAME}\t{rs/n*100:.3f}\t{rp/n*100:.3f}\t{n}\n")
PYEOF2
    else
        echo "  -> All runs failed."
        echo -e "${EXP_NAME}\tN/A\tN/A\t0" >> "${RESULTS_DIR}/summary.txt"
    fi
}

echo -e "Experiment\tATE_RMSE(cm)\tRPE_RMSE(cm)\tValid_Runs" > "${RESULTS_DIR}/summary.txt"


# ══════════════════════════════════════════════════════════════
# PART 0: Gray Baseline（参考）
# ══════════════════════════════════════════════════════════════
cat > /tmp/snip.yml <<'EOF'
tracking:
  color: gray
mapping:
  color: gray
EOF
run_experiment "Gray_Baseline" /tmp/snip.yml


# ══════════════════════════════════════════════════════════════
# PART 1: Conv1 跳变节点分析
# ══════════════════════════════════════════════════════════════

# [类型A] Ch54 单独评估（s5→s6 加入后 ATE 8.18→6.78，改善最大）
cat > /tmp/snip.yml <<'EOF'
tracking:
  color: cnn
  cnn_mode: cnn_only
  cnn_layer_name: conv1
  cnn_channels: 1
  cnn_channel_select: "d54"
  cnn_layer_full_channels: 64
mapping:
  color: gray
EOF
run_experiment "Conv1_Single_Ch54" /tmp/snip.yml

# [类型A] Ch54 + 当前 s6 最优 subset（验证 Ch54 是否是 s6 中的核心贡献者）
# s6 最优 = [6,28,34,62,12,54]，已知结果 6.779cm
# 这里测试 Ch54 与 s4 最优 [6,28,34,62] 的组合（s4 是 BQS elbow，5通道）
cat > /tmp/snip.yml <<'EOF'
tracking:
  color: cnn
  cnn_mode: cnn_only
  cnn_layer_name: conv1
  cnn_channels: 5
  cnn_channel_select: "d6,d28,d34,d62,d54"
  cnn_layer_full_channels: 64
mapping:
  color: gray
EOF
run_experiment "Conv1_BQSElbow_plus_Ch54" /tmp/snip.yml

# [类型B] 剔除 Ch34（s2→s3 加入后 ATE 恶化 +1.56cm）
# 从 s6 最优 [6,28,34,62,12,54] 中移除 Ch34 → [6,28,62,12,54]
cat > /tmp/snip.yml <<'EOF'
tracking:
  color: cnn
  cnn_mode: cnn_only
  cnn_layer_name: conv1
  cnn_channels: 5
  cnn_channel_select: "d6,d28,d62,d12,d54"
  cnn_layer_full_channels: 64
mapping:
  color: gray
EOF
run_experiment "Conv1_Best6_minus_Ch34" /tmp/snip.yml

# [类型B] 剔除 Ch03（s6→s7 加入后 ATE 恶化 +1.96cm）
# s7 = [6,28,34,62,12,54,3]，已知 8.738cm；去掉 Ch03 即 s6，已知 6.779cm
# 这里额外验证：s6 最优 + 替换 Ch34 为 Ch03（即 [6,28,62,12,54,3]）
cat > /tmp/snip.yml <<'EOF'
tracking:
  color: cnn
  cnn_mode: cnn_only
  cnn_layer_name: conv1
  cnn_channels: 6
  cnn_channel_select: "d6,d28,d62,d12,d54,d3"
  cnn_layer_full_channels: 64
mapping:
  color: gray
EOF
run_experiment "Conv1_Best6_minus_Ch34_plus_Ch03" /tmp/snip.yml


# ══════════════════════════════════════════════════════════════
# PART 2: Layer1 跳变节点分析
# ══════════════════════════════════════════════════════════════

# [类型A] Ch61 单独评估（s1→s2 加入后 ATE 15.88→7.32，改善 -8.56cm，最大跳变）
cat > /tmp/snip.yml <<'EOF'
tracking:
  color: cnn
  cnn_mode: cnn_only
  cnn_layer_name: layer1
  cnn_channels: 1
  cnn_channel_select: "d61"
  cnn_layer_full_channels: 64
mapping:
  color: gray
EOF
run_experiment "Layer1_Single_Ch61" /tmp/snip.yml

# [类型A] Ch41 单独评估（s5→s6 加入后 ATE 6.76→5.91，改善 -0.85cm，全局最优关键通道）
cat > /tmp/snip.yml <<'EOF'
tracking:
  color: cnn
  cnn_mode: cnn_only
  cnn_layer_name: layer1
  cnn_channels: 1
  cnn_channel_select: "d41"
  cnn_layer_full_channels: 64
mapping:
  color: gray
EOF
run_experiment "Layer1_Single_Ch41" /tmp/snip.yml

# [类型A] Ch41 + BQS Elbow subset（s5 = [2,61,60,32,53]，BQS most cost-effective）
# 验证 Ch41 加入 BQS elbow subset 后是否也能达到全局最优
cat > /tmp/snip.yml <<'EOF'
tracking:
  color: cnn
  cnn_mode: cnn_only
  cnn_layer_name: layer1
  cnn_channels: 6
  cnn_channel_select: "d2,d61,d60,d32,d53,d41"
  cnn_layer_full_channels: 64
mapping:
  color: gray
EOF
run_experiment "Layer1_BQSElbow_plus_Ch41" /tmp/snip.yml

# [类型B] 剔除 Ch32（s3→s4 加入后 ATE 轻微恶化 7.11→7.25）
# 从 s6 最优 [2,61,60,32,53,41] 中移除 Ch32 → [2,61,60,53,41]
cat > /tmp/snip.yml <<'EOF'
tracking:
  color: cnn
  cnn_mode: cnn_only
  cnn_layer_name: layer1
  cnn_channels: 5
  cnn_channel_select: "d2,d61,d60,d53,d41"
  cnn_layer_full_channels: 64
mapping:
  color: gray
EOF
run_experiment "Layer1_Best6_minus_Ch32" /tmp/snip.yml

# [类型A] Ch61+Ch41 组合（两个最大正贡献通道，跳过其他通道）
cat > /tmp/snip.yml <<'EOF'
tracking:
  color: cnn
  cnn_mode: cnn_only
  cnn_layer_name: layer1
  cnn_channels: 2
  cnn_channel_select: "d61,d41"
  cnn_layer_full_channels: 64
mapping:
  color: gray
EOF
run_experiment "Layer1_Ch61_Ch41_only" /tmp/snip.yml


# ══════════════════════════════════════════════════════════════
# PART 3: Layer2 跳变节点分析
# ══════════════════════════════════════════════════════════════

# [类型A] Ch66 单独评估（s1→s2 加入后 ATE 31.65→6.23，改善 -25.42cm，最大跳变）
cat > /tmp/snip.yml <<'EOF'
tracking:
  color: cnn
  cnn_mode: cnn_only
  cnn_layer_name: layer2
  cnn_channels: 1
  cnn_channel_select: "d66"
  cnn_layer_full_channels: 128
mapping:
  color: gray
EOF
run_experiment "Layer2_Single_Ch66" /tmp/snip.yml

# [类型B] 剔除 Ch39（s2→s3 加入后 ATE 6.23→15.91，恶化 +9.68cm，最大负跳变）
# s2 最优 [120,66] = 6.234cm（已知），这里验证 Ch39 单独评估
cat > /tmp/snip.yml <<'EOF'
tracking:
  color: cnn
  cnn_mode: cnn_only
  cnn_layer_name: layer2
  cnn_channels: 1
  cnn_channel_select: "d39"
  cnn_layer_full_channels: 128
mapping:
  color: gray
EOF
run_experiment "Layer2_Single_Ch39" /tmp/snip.yml

# [类型B] [120,66] + 替换 Ch39 为 Layer2 Top-1 BQS 通道 Ch039（即 d39，同一个）
# 注意：Layer2 BQS Top-1 是 Ch039，BQS=0.6745，但 ATE 中加入后恶化
# 这里测试 [120,66] + BQS Top-4 Ch058（BQS=0.6322，未出现在 Greedy 路径中）
cat > /tmp/snip.yml <<'EOF'
tracking:
  color: cnn
  cnn_mode: cnn_only
  cnn_layer_name: layer2
  cnn_channels: 3
  cnn_channel_select: "d120,d66,d58"
  cnn_layer_full_channels: 128
mapping:
  color: gray
EOF
run_experiment "Layer2_Best2_plus_Ch58" /tmp/snip.yml

# [类型A] Ch66 + BQS Top-1 Ch039（验证 BQS 最优组合在 ATE 上的表现）
cat > /tmp/snip.yml <<'EOF'
tracking:
  color: cnn
  cnn_mode: cnn_only
  cnn_layer_name: layer2
  cnn_channels: 2
  cnn_channel_select: "d66,d39"
  cnn_layer_full_channels: 128
mapping:
  color: gray
EOF
run_experiment "Layer2_Ch66_Ch39_only" /tmp/snip.yml


# ══════════════════════════════════════════════════════════════
# 最终汇总
# ══════════════════════════════════════════════════════════════
echo ""
echo "=========================================="
echo "FINAL SUMMARY"
echo "=========================================="
echo ""
echo "Reference (from ablation):"
echo "  Gray_Baseline:        7.736 cm"
echo "  Conv1 s6 best:        6.779 cm  [6,28,34,62,12,54]"
echo "  Layer1 s6 best:       5.908 cm  [2,61,60,32,53,41]"
echo "  Layer2 s2 best:       6.234 cm  [120,66]"
echo ""
echo "New experiments:"
column -t -s $'\t' "${RESULTS_DIR}/summary.txt"
echo ""
echo "All done at $(date)"