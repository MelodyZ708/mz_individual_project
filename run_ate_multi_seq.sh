#!/bin/bash
#SBATCH --job-name=como_multi_seq
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=20:00:00
#SBATCH --output=/vol/bitbucket/mz325/individual_project/logs/multi_seq_%j.out
#SBATCH --error=/vol/bitbucket/mz325/individual_project/logs/multi_seq_%j.err

echo "=========================================="
echo "Multi-Sequence ATE Evaluation"
echo "Configs: gray | coarse | fine | c2f-a | c2f-b"
echo "Datasets: fr2/desk, fr3/long_office (clean + flashlight + lightswitch)"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=========================================="

eval "$(/vol/bitbucket/mz325/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate /vol/bitbucket/mz325/envs/como

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
PROJECT_DIR="/vol/bitbucket/mz325/individual_project/como"
CONFIG_FILE="${PROJECT_DIR}/config/como.yml"
CONFIG_BACKUP="${PROJECT_DIR}/config/como.yml.bak"
RESULTS_DIR="${PROJECT_DIR}/results/multi_seq_eval"
LOG_DIR="/vol/bitbucket/mz325/individual_project/logs"
DATASET_ROOT="/vol/bitbucket/mz325/datasets/tum"

mkdir -p "${RESULTS_DIR}"
mkdir -p "${LOG_DIR}"

# ─────────────────────────────────────────────
# Datasets (name : relative path under DATASET_ROOT)
# ─────────────────────────────────────────────
declare -A DATASETS
DATASETS["fr2_desk_clean"]="rgbd_dataset_freiburg2_desk"
DATASETS["fr2_desk_flashlight"]="rgbd_dataset_freiburg2_desk_flashlight"
DATASETS["fr2_desk_lightswitch"]="rgbd_dataset_freiburg2_desk_lightswitch"
DATASETS["fr3_office_clean"]="rgbd_dataset_freiburg3_long_office_household"
DATASETS["fr3_office_flashlight"]="rgbd_dataset_freiburg3_long_office_household_flashlight"
DATASETS["fr3_office_lightswitch"]="rgbd_dataset_freiburg3_long_office_household_lightswitch"

# Ordered list for iteration
DATASET_KEYS=(
    "fr2_desk_clean"
    "fr2_desk_flashlight"
    "fr2_desk_lightswitch"
    "fr3_office_clean"
    "fr3_office_flashlight"
    "fr3_office_lightswitch"
)

# ─────────────────────────────────────────────
# Experiment configs
# c2f_c removed: 3-level pyramid with fine_levels=3 → 0 coarse levels → invalid
# ─────────────────────────────────────────────
CONFIGS=(
    "gray"
    "coarse_only"
    "fine_only"
    "c2f_a"
    "c2f_b"
)

write_snippet() {
    local EXP="$1"
    local SNIP="/tmp/snip_${EXP}.yml"

    case "${EXP}" in
    gray)
        cat > "${SNIP}" <<'EOF'
tracking:
  color: gray
  cnn_mode: cnn_only
EOF
        ;;
    coarse_only)
        cat > "${SNIP}" <<'EOF'
tracking:
  color: cnn
  cnn_layer_name: layer2
  cnn_channels: 3
  cnn_channel_select: "d120,d66,d54"
  cnn_layer_full_channels: 128
  cnn_mode: cnn_only
EOF
        ;;
    fine_only)
        cat > "${SNIP}" <<'EOF'
tracking:
  color: cnn
  cnn_layer_name: conv1
  cnn_channels: 6
  cnn_channel_select: "d6,d28,d34,d50,d39,d16"
  cnn_layer_full_channels: 64
  cnn_mode: cnn_only
EOF
        ;;
    c2f_a)
        cat > "${SNIP}" <<'EOF'
tracking:
  color: cnn_c2f
  cnn_c2f_version: A
  cnn_layer_coarse: layer2
  cnn_channels_coarse: 3
  cnn_channel_select_coarse: "d120,d66,d54"
  cnn_layer_fine: conv1
  cnn_channels_fine: 6
  cnn_channel_select_fine: "d6,d28,d34,d50,d39,d16"
  cnn_layer_full_channels: 64
  cnn_mode: cnn_only
EOF
        ;;
    c2f_b)
        cat > "${SNIP}" <<'EOF'
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
        ;;
    esac
}

# ─────────────────────────────────────────────
# Apply snippet to como.yml
# ─────────────────────────────────────────────
apply_snippet() {
    local EXP="$1"
    local SNIP="/tmp/snip_${EXP}.yml"

    python3 - <<PYEOF
import yaml, sys

config_file = "${CONFIG_FILE}"
snip_file   = "${SNIP}"

with open(config_file, "r") as f:
    cfg = yaml.safe_load(f)
with open(snip_file, "r") as f:
    snippet = yaml.safe_load(f)

if "tracking" in snippet:
    cfg["tracking"].update(snippet["tracking"])

with open(config_file, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

color = cfg["tracking"].get("color", "?")
c2f_v = cfg["tracking"].get("cnn_c2f_version", "")
tag = f"C2F-{c2f_v}" if color == "cnn_c2f" else color
print(f"[Config] {tag} applied for experiment: ${EXP}")
PYEOF
}

# ─────────────────────────────────────────────
# Single run + ATE evaluation
# ─────────────────────────────────────────────
run_one() {
    local EXP="$1"
    local DS_KEY="$2"
    local DS_NAME="${DATASETS[${DS_KEY}]}"
    local DS_PATH="${DATASET_ROOT}/${DS_NAME}"
    local GT_FILE="${DS_PATH}/groundtruth.txt"
    local SAVE_NAME="${EXP}__${DS_KEY}"
    local TRAJ_SAVED="${RESULTS_DIR}/${SAVE_NAME}.txt"
    local SUMMARY_FILE="${RESULTS_DIR}/summary.tsv"

    # COMO saves trajectory to ./results/tum_<dataset_name>.txt
    # derived from: seq_path.rstrip("/").rsplit("/", 3) → tmp[1]+"_"+tmp[2]
    # e.g. /vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg2_desk
    #   → tmp[1]="tum", tmp[2]="rgbd_dataset_freiburg2_desk"
    #   → tum_rgbd_dataset_freiburg2_desk.txt
    local TRAJ_FILE="${PROJECT_DIR}/results/datasets_tum.txt"

    echo ""
    echo "##################################################"
    echo "# EXP: ${EXP}  |  DATASET: ${DS_KEY}"
    echo "##################################################"

    # Write + apply snippet
    write_snippet "${EXP}"
    apply_snippet "${EXP}"

    # Remove stale trajectory
    rm -f "${TRAJ_FILE}"

    # Virtual display
    local DISP=$(( 400 + RANDOM % 100 ))
    Xvfb :${DISP} -screen 0 1920x1080x24 &
    local XVFB_PID=$!
    sleep 3
    export DISPLAY=:${DISP}

    timeout 900 python como/como_dataset.py \
        --dataset_type=tum \
        --dataset_dir="${DS_PATH}" \
        || true

    kill ${XVFB_PID} 2>/dev/null

    # ── Check trajectory ──────────────────────────────────────────────────────
    if [ ! -f "${TRAJ_FILE}" ]; then
        STATUS="FAIL"
        ATE_MEAN="-"
        echo "  [FAIL] Trajectory not found — COMO crashed or timed out."
        echo "  Expected: ${TRAJ_FILE}"
        echo -e "${DS_KEY}\t${EXP}\t${ATE_MEAN}\t${STATUS}" >> "${SUMMARY_FILE}"
        return
    fi

    # ── NaN guard ─────────────────────────────────────────────────────────────
    if grep -qi "nan" "${TRAJ_FILE}"; then
        STATUS="NAN_FAIL"
        ATE_MEAN="-"
        echo "  [FAIL] Trajectory contains NaN."
        cp "${TRAJ_FILE}" "${TRAJ_SAVED}.nan"
        echo -e "${DS_KEY}\t${EXP}\t${ATE_MEAN}\t${STATUS}" >> "${SUMMARY_FILE}"
        return
    fi

    cp "${TRAJ_FILE}" "${TRAJ_SAVED}"

    # ── ATE Mean (cm) ─────────────────────────────────────────────────────────
    ATE_MEAN=$(evo_ape tum "${GT_FILE}" "${TRAJ_SAVED}" \
        --align --correct_scale 2>/dev/null \
        | grep -w "mean" | awk '{printf "%.4f", $2 * 100}')

    if [ -z "${ATE_MEAN}" ]; then
        STATUS="PARSE_FAIL"
        ATE_MEAN="-"
    else
        STATUS="OK"
    fi

    echo "  ATE Mean = ${ATE_MEAN} cm  |  ${STATUS}"
    echo -e "${DS_KEY}\t${EXP}\t${ATE_MEAN}\t${STATUS}" >> "${SUMMARY_FILE}"
}

# ─────────────────────────────────────────────
# Main: backup config, iterate all combos
# ─────────────────────────────────────────────
cp "${CONFIG_FILE}" "${CONFIG_BACKUP}"
trap "echo '[Cleanup] Restoring original config...'; cp '${CONFIG_BACKUP}' '${CONFIG_FILE}'" EXIT

cd "${PROJECT_DIR}"

# Write TSV header
SUMMARY_FILE="${RESULTS_DIR}/summary.tsv"
echo -e "Dataset\tExperiment\tATE_Mean_cm\tStatus" > "${SUMMARY_FILE}"

for DS_KEY in "${DATASET_KEYS[@]}"; do
    for EXP in "${CONFIGS[@]}"; do
        run_one "${EXP}" "${DS_KEY}"
    done
done

# ─────────────────────────────────────────────
# Print final summary table
# ─────────────────────────────────────────────
echo ""
echo "=========================================="
echo "FINAL SUMMARY"
echo "=========================================="
echo ""

for DS_KEY in "${DATASET_KEYS[@]}"; do
    echo "  Dataset: ${DS_KEY}"
    printf "  %-22s  %14s  %8s\n" "Experiment" "ATE Mean(cm)" "Status"
    printf "  %-22s  %14s  %8s\n" "----------------------" "--------------" "--------"
    grep "^${DS_KEY}" "${SUMMARY_FILE}" | while IFS=$'\t' read -r ds exp ate_m stat; do
        printf "  %-22s  %14s  %8s\n" "${exp}" "${ate_m}" "${stat}"
    done
    echo ""
done

echo "Full results saved to: ${SUMMARY_FILE}"
echo "Trajectory files in:   ${RESULTS_DIR}/"
echo "All done at $(date)"
echo "=========================================="