#!/bin/bash
echo "=========================================="
echo "Multi-Sequence ATE Evaluation — fr2/fr3"
echo "Host: $(hostname) | Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "=========================================="

PROJECT_DIR="/home/melody/code/individual_project/como"
CONFIG_FILE="${PROJECT_DIR}/config/como.yml"
CONFIG_BACKUP="${PROJECT_DIR}/config/como.yml.bak"
TRAJ_FILE="${PROJECT_DIR}/results/data_tum.txt"
RESULTS_DIR="${PROJECT_DIR}/results/multi_seq_eval_fr2fr3"
DATASET_ROOT="/home/melody/data/tum"

mkdir -p "${RESULTS_DIR}" "${PROJECT_DIR}/logs"

# ── Datasets ────────────────────────────────────────────────────────────────
declare -A DATASETS
DATASETS["fr2_desk_clean"]="rgbd_dataset_freiburg2_desk"
DATASETS["fr2_desk_flashlight"]="rgbd_dataset_freiburg2_desk_flashlight"
DATASETS["fr2_desk_lightswitch"]="rgbd_dataset_freiburg2_desk_lightswitch"
DATASETS["fr3_office_clean"]="rgbd_dataset_freiburg3_long_office_household"
DATASETS["fr3_office_flashlight"]="rgbd_dataset_freiburg3_long_office_household_flashlight"
DATASETS["fr3_office_lightswitch"]="rgbd_dataset_freiburg3_long_office_household_lightswitch"

DATASET_KEYS=("fr2_desk_clean" "fr2_desk_flashlight" "fr2_desk_lightswitch"
              "fr3_office_clean" "fr3_office_flashlight" "fr3_office_lightswitch")

# c2f_c excluded: fine_levels=3 → 0 coarse levels → code validation error
CONFIGS=("gray" "coarse_only" "fine_only" "c2f_a" "c2f_b")

# ── Config snippets ──────────────────────────────────────────────────────────
write_snippet() {
    local EXP="$1"; local SNIP="/tmp/snip_${EXP}.yml"
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

apply_snippet() {
    local EXP="$1"; local SNIP="/tmp/snip_${EXP}.yml"
    python3 - <<PYEOF
import yaml
cfg  = yaml.safe_load(open("${CONFIG_FILE}"))
snip = yaml.safe_load(open("${SNIP}"))
if "tracking" in snip:
    cfg["tracking"].update(snip["tracking"])
yaml.dump(cfg, open("${CONFIG_FILE}", "w"), default_flow_style=False, allow_unicode=True)
color = cfg["tracking"].get("color","?")
v     = cfg["tracking"].get("cnn_c2f_version","")
tag   = f"C2F-{v}" if color=="cnn_c2f" else color
print(f"[Config] {tag} applied for experiment: ${EXP}")
PYEOF
}

# ── Run one experiment ───────────────────────────────────────────────────────
run_one() {
    local EXP="$1"; local DS_KEY="$2"
    local DS_PATH="${DATASET_ROOT}/${DATASETS[${DS_KEY}]}"
    local GT_FILE="${DS_PATH}/groundtruth.txt"
    local TRAJ_SAVED="${RESULTS_DIR}/${EXP}__${DS_KEY}.txt"
    local SUMMARY_FILE="${RESULTS_DIR}/summary.tsv"

    echo ""
    echo "##################################################"
    echo "# EXP: ${EXP}  |  DATASET: ${DS_KEY}"
    echo "##################################################"

    write_snippet "${EXP}"
    apply_snippet "${EXP}"
    rm -f "${TRAJ_FILE}"

    timeout 300 python como/como_dataset.py \
        --dataset_type=tum --dataset_dir="${DS_PATH}" || true

    if [ ! -f "${TRAJ_FILE}" ]; then
        STATUS="FAIL"; ATE_MEAN="-"
        echo "  [FAIL] Trajectory file not produced."
        echo -e "${DS_KEY}\t${EXP}\t${ATE_MEAN}\t${STATUS}" >> "${SUMMARY_FILE}"
        return
    fi

    # ── NaN guard: if trajectory contains nan, mark as FAIL ─────────────────
    if grep -qi "nan" "${TRAJ_FILE}"; then
        STATUS="FAIL"; ATE_MEAN="-"
        echo "  [FAIL] Trajectory contains NaN values."
        cp "${TRAJ_FILE}" "${TRAJ_SAVED}.nan"
        echo -e "${DS_KEY}\t${EXP}\t${ATE_MEAN}\t${STATUS}" >> "${SUMMARY_FILE}"
        return
    fi

    cp "${TRAJ_FILE}" "${TRAJ_SAVED}"

    ATE_MEAN=$(evo_ape tum "${GT_FILE}" "${TRAJ_SAVED}" \
        --align --correct_scale 2>/dev/null \
        | grep -w "mean" | awk '{printf "%.4f", $2*100}')

    if [ -z "${ATE_MEAN}" ]; then
        STATUS="PARSE_FAIL"; ATE_MEAN="-"
    else
        STATUS="OK"
    fi

    echo "  ATE Mean = ${ATE_MEAN} cm  |  ${STATUS}"
    echo -e "${DS_KEY}\t${EXP}\t${ATE_MEAN}\t${STATUS}" >> "${SUMMARY_FILE}"
}

# ── Main ─────────────────────────────────────────────────────────────────────
cp "${CONFIG_FILE}" "${CONFIG_BACKUP}"
trap "echo '[Cleanup] Restoring config...'; cp '${CONFIG_BACKUP}' '${CONFIG_FILE}'" EXIT
cd "${PROJECT_DIR}"

SUMMARY_FILE="${RESULTS_DIR}/summary.tsv"
echo -e "Dataset\tExperiment\tATE_Mean_cm\tStatus" > "${SUMMARY_FILE}"

for DS_KEY in "${DATASET_KEYS[@]}"; do
    for EXP in "${CONFIGS[@]}"; do
        run_one "${EXP}" "${DS_KEY}"
    done
done

# ── Final summary ─────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "FINAL SUMMARY — ATE Mean (cm)"
echo "=========================================="
for DS_KEY in "${DATASET_KEYS[@]}"; do
    echo "  Dataset: ${DS_KEY}"
    printf "  %-22s  %14s  %8s\n" "Experiment" "ATE Mean (cm)" "Status"
    grep "^${DS_KEY}" "${SUMMARY_FILE}" | \
        while IFS=$'\t' read -r ds exp ate_m stat; do
            printf "  %-22s  %14s  %8s\n" "${exp}" "${ate_m}" "${stat}"
        done
    echo ""
done
echo "All done at $(date)"