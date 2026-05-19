#!/bin/bash
# run_experiment.sh — 后台运行 CoMo 实验，自动记录配置和输出
#
# 用法:
#   ./run_experiment.sh                          # 前台运行
#   nohup ./run_experiment.sh > /dev/null 2>&1 & # 后台运行

set -e

# ── 配置 ──
DATASET_DIR="/home/melody/data/tum/rgbd_dataset_freiburg2_desk/"
DATASET_TYPE="tum"
NUM_RUNS=3
PROJECT_DIR="$HOME/code/individual_project/como"

# ── 自动生成带时间戳的日志目录 ──
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT_DIR}/logs/${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

# ── 记录实验信息 ──
cd "${PROJECT_DIR}"

echo "=== Experiment: ${TIMESTAMP} ===" | tee "${LOG_DIR}/run.log"
echo "Host: $(hostname)" | tee -a "${LOG_DIR}/run.log"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)" | tee -a "${LOG_DIR}/run.log"
echo "Dataset: ${DATASET_DIR}" | tee -a "${LOG_DIR}/run.log"
echo "Num runs: ${NUM_RUNS}" | tee -a "${LOG_DIR}/run.log"
echo "" | tee -a "${LOG_DIR}/run.log"

# 保存当前配置快照
cp config/como.yml "${LOG_DIR}/como.yml.snapshot"
echo "Config snapshot saved to ${LOG_DIR}/como.yml.snapshot"

# ── 运行 ──
for RUN in $(seq 1 ${NUM_RUNS}); do
    echo "" | tee -a "${LOG_DIR}/run.log"
    echo "=== Run ${RUN}/${NUM_RUNS} started at $(date) ===" | tee -a "${LOG_DIR}/run.log"
    
    python como/como_dataset.py \
        --dataset_type=${DATASET_TYPE} \
        --dataset_dir=${DATASET_DIR} \
        2>&1 | tee "${LOG_DIR}/run${RUN}_output.log"
    
    # 保存轨迹文件
    TRAJ_FILE=$(ls -t results/*.txt 2>/dev/null | head -1)
    if [ -n "$TRAJ_FILE" ]; then
        cp "$TRAJ_FILE" "${LOG_DIR}/trajectory_run${RUN}.txt"
        echo "Trajectory saved: trajectory_run${RUN}.txt" | tee -a "${LOG_DIR}/run.log"
    else
        echo "[WARN] Run ${RUN}: No trajectory file found" | tee -a "${LOG_DIR}/run.log"
    fi
    
    echo "=== Run ${RUN}/${NUM_RUNS} finished at $(date) ===" | tee -a "${LOG_DIR}/run.log"
done

echo ""
echo "=== All done at $(date) ===" | tee -a "${LOG_DIR}/run.log"
echo "Results in: ${LOG_DIR}" | tee -a "${LOG_DIR}/run.log"
