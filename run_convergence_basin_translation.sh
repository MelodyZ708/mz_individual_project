#!/bin/bash
#SBATCH --job-name=basin_trans
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/vol/bitbucket/mz325/individual_project/logs/basin_translation_%j.out
#SBATCH --error=/vol/bitbucket/mz325/individual_project/logs/basin_translation_%j.err

# ============================================================
# Convergence Basin — Translation (Clean / +30% / +50%)
# One-shot script: runs all 3 brightness conditions in a single job
# ============================================================

PROJECT_DIR="/vol/bitbucket/mz325/individual_project"
CONFIG_FILE="${PROJECT_DIR}/como/config/como.yml"
OUTPUT_DIR="${PROJECT_DIR}/vis_results/convergence_basin_translation"
LOG_DIR="${PROJECT_DIR}/logs"

echo "=========================================="
echo "  Convergence Basin — Translation"
echo "  Conditions: Clean / +30% / +50%"
echo "=========================================="
echo "  Node:    $(hostname)"
echo "  GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Date:    $(date)"
echo "  Project: ${PROJECT_DIR}"
echo "  Output:  ${OUTPUT_DIR}"
echo "=========================================="

# ── Environment setup ──
eval "$(/vol/bitbucket/mz325/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate /vol/bitbucket/mz325/envs/como

# Virtual display (needed for matplotlib Agg backend on headless nodes)
Xvfb :204 -screen 0 1920x1080x24 &
XVFB_PID=$!
sleep 1
export DISPLAY=:204

# Cleanup on exit (restore config + kill Xvfb)
cleanup() {
    if [[ -f "${CONFIG_FILE}.bak_basin" ]]; then
        cp "${CONFIG_FILE}.bak_basin" "$CONFIG_FILE"
        rm -f "${CONFIG_FILE}.bak_basin"
        echo "--- Config restored ---"
    fi
    kill $XVFB_PID 2>/dev/null || true
}
trap cleanup EXIT

cd "$PROJECT_DIR"

# ── Ensure config uses conv1 ──
echo ""
echo "--- Ensuring cnn_layer: conv1 ---"
cp "$CONFIG_FILE" "${CONFIG_FILE}.bak_basin"
sed -i 's/cnn_layer: layer1/cnn_layer: conv1/' "$CONFIG_FILE"
grep "cnn_layer" "$CONFIG_FILE"

# ── Create output directories ──
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# ── Run ──
echo ""
echo "--- Running convergence basin (translation) ---"
echo "  Grid:       61x61 (3721 evaluations per modality per condition)"
echo "  Shift:      +/-30 px"
echo "  Conditions: Clean, Brightness +30%, Brightness +50%"
echo "  Frames:     3 (early / middle / late)"
echo "  Sharpness:  +/-5 px around minimum"
echo ""

python visualize_convergence_basin_translation.py

echo ""
echo "=========================================="
echo "  DONE at $(date)"
echo "  Output: ${OUTPUT_DIR}/"
echo "=========================================="