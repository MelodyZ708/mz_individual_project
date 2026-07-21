
#!/bin/bash
# ============================================================
# Step 4: C2F Channel Optimization Experiments
#
# Configurations (9 total):
#   gray                — pure grayscale baseline
#   fine_only           — conv1 BestATE channels only (cnn_only)
#   coarse_only         — layer2 BestATE channels only (cnn_only)
#   c2f_a_BestATE       — C2F version A, BestATE channels
#   c2f_a_FreqDerived   — C2F version A, FreqDerived channels
#   c2f_a_PrincipleBased— C2F version A, PrincipleBased channels
#   c2f_b_BestATE       — C2F version B, BestATE channels
#   c2f_b_FreqDerived   — C2F version B, FreqDerived channels
#   c2f_b_PrincipleBased— C2F version B, PrincipleBased channels
#
# Sequences (9 total):
#   fr1/fr2/fr3 × clean / flashlight / lightswitch
#
# Total runs: 9 configs × 9 sequences = 81
# Timeout per run: 900s
#
# Failure detection:
#   1. FAIL         — trajectory file not produced (crash or timeout)
#   2. NAN_FAIL     — trajectory contains NaN values
#   3. PARSE_FAIL   — evo_ape output could not be parsed
# ============================================================

echo "=========================================="
echo "Step 4: C2F Channel Optimization Experiments"
echo "Host: $(hostname) | Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "=========================================="

# ── Activate conda ─────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate como

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_DIR="/home/melody/code/individual_project/como"
CONFIG_FILE="${PROJECT_DIR}/config/como.yml"
CONFIG_BACKUP="${PROJECT_DIR}/config/como.yml.bak"
TRAJ_FILE="${PROJECT_DIR}/results/data_tum.txt"
RESULTS_DIR="${PROJECT_DIR}/results/step4_c2f_experiments"
DATASET_ROOT="/home/melody/data/tum"
SUMMARY_FILE="${RESULTS_DIR}/summary.tsv"

mkdir -p "${RESULTS_DIR}"

# ── Datasets ───────────────────────────────────────────────────────────────────
declare -A DATASETS
DATASETS["fr1_clean"]="rgbd_dataset_freiburg1_desk"
DATASETS["fr1_flashlight"]="rgbd_dataset_freiburg1_desk_flashlight"
DATASETS["fr1_lightswitch"]="rgbd_dataset_freiburg1_desk_lightswitch"
DATASETS["fr2_clean"]="rgbd_dataset_freiburg2_desk"
DATASETS["fr2_flashlight"]="rgbd_dataset_freiburg2_desk_flashlight"
DATASETS["fr2_lightswitch"]="rgbd_dataset_freiburg2_desk_lightswitch"
DATASETS["fr3_clean"]="rgbd_dataset_freiburg3_long_office_household"
DATASETS["fr3_flashlight"]="rgbd_dataset_freiburg3_long_office_household_flashlight"
DATASETS["fr3_lightswitch"]="rgbd_dataset_freiburg3_long_office_household_lightswitch"

DATASET_KEYS=(
    "fr1_clean"
    "fr1_flashlight"
    "fr1_lightswitch"
    "fr2_clean"
    "fr2_flashlight"
    "fr2_lightswitch"
    "fr3_clean"
    "fr3_flashlight"
    "fr3_lightswitch"
)

# ── Experiment configs ─────────────────────────────────────────────────────────
CONFIGS=(
    "gray"
    "fine_only"
    "coarse_only"
    "c2f_a_BestATE"
    "c2f_a_FreqDerived"
    "c2f_a_PrincipleBased"
    "c2f_b_BestATE"
    "c2f_b_FreqDerived"
    "c2f_b_PrincipleBased"
)

# ── Write YAML snippet for each config ────────────────────────────────────────
write_snippet() {
    local EXP="$1"
    local SNIP="/tmp/snip_${EXP}.yml"

    case "${EXP}" in

    # ------------------------------------------------------------------
    # Baseline: pure grayscale
    # ------------------------------------------------------------------
    gray)
        cat > "${SNIP}" <<'EOF'
tracking:
  color: gray
EOF
        ;;

    # ------------------------------------------------------------------
    # fine_only: conv1 BestATE channels, cnn_only (no C2F)
    # Channels: d5,d29,d40,d52  (Top1 random search, ATE 15.2 cm)
    # ------------------------------------------------------------------
    fine_only)
        cat > "${SNIP}" <<'EOF'
tracking:
  color: cnn
  cnn_mode: cnn_only
  cnn_layer_name: conv1
  cnn_channels: 4
  cnn_channel_select: "d5,d29,d40,d52"
  cnn_layer_full_channels: 64
EOF
        ;;

    # ------------------------------------------------------------------
    # coarse_only: layer2 BestATE channels, cnn_only (no C2F)
    # Channels: d7,d37,d60,d67,d74,d104  (Top1 random search, ATE 21.9 cm)
    # ------------------------------------------------------------------
    coarse_only)
        cat > "${SNIP}" <<'EOF'
tracking:
  color: cnn
  cnn_mode: cnn_only
  cnn_layer_name: layer2
  cnn_channels: 6
  cnn_channel_select: "d7,d37,d60,d67,d74,d104"
  cnn_layer_full_channels: 128
EOF
        ;;

    # ------------------------------------------------------------------
    # C2F version A — BestATE channels
    # Fine  (conv1):  d5,d29,d40,d52   (Top1 random search)
    # Coarse (layer2): d7,d37,d60,d67,d74,d104  (Top1 random search)
    # ------------------------------------------------------------------
    c2f_a_BestATE)
        cat > "${SNIP}" <<'EOF'
tracking:
  color: cnn_c2f
  cnn_c2f_version: A
  cnn_layer_fine: conv1
  cnn_channels_fine: 4
  cnn_channel_select_fine: "d5,d29,d40,d52"
  cnn_layer_full_channels_fine: 64
  cnn_layer_coarse: layer2
  cnn_channels_coarse: 6
  cnn_channel_select_coarse: "d7,d37,d60,d67,d74,d104"
  cnn_layer_full_channels_coarse: 128
  cnn_mode: cnn_only
EOF
        ;;

    # ------------------------------------------------------------------
    # C2F version A — FreqDerived channels
    # Fine  (conv1):  d5,d25,d22,d6    (top-30 high-frequency channels)
    # Coarse (layer2): d121,d67,d60,d51 (top-50 high-frequency channels)
    # ------------------------------------------------------------------
    c2f_a_FreqDerived)
        cat > "${SNIP}" <<'EOF'
tracking:
  color: cnn_c2f
  cnn_c2f_version: A
  cnn_layer_fine: conv1
  cnn_channels_fine: 4
  cnn_channel_select_fine: "d5,d25,d22,d6"
  cnn_layer_full_channels_fine: 64
  cnn_layer_coarse: layer2
  cnn_channels_coarse: 4
  cnn_channel_select_coarse: "d121,d67,d60,d51"
  cnn_layer_full_channels_coarse: 128
  cnn_mode: cnn_only
EOF
        ;;

    # ------------------------------------------------------------------
    # C2F version A — PrincipleBased channels (Step 2 principle-driven)
    # Fine  (conv1):  d5,d6,d52,d15    (low mean-diff edge-anchor channels)
    # Coarse (layer2): d51,d60,d67,d9  (ultra-sparse anchor + coarse-region)
    # ------------------------------------------------------------------
    c2f_a_PrincipleBased)
        cat > "${SNIP}" <<'EOF'
tracking:
  color: cnn_c2f
  cnn_c2f_version: A
  cnn_layer_fine: conv1
  cnn_channels_fine: 4
  cnn_channel_select_fine: "d5,d6,d52,d15"
  cnn_layer_full_channels_fine: 64
  cnn_layer_coarse: layer2
  cnn_channels_coarse: 4
  cnn_channel_select_coarse: "d51,d60,d67,d9"
  cnn_layer_full_channels_coarse: 128
  cnn_mode: cnn_only
EOF
        ;;

    # ------------------------------------------------------------------
    # C2F version B — BestATE channels (same channels as A, different arch)
    # ------------------------------------------------------------------
    c2f_b_BestATE)
        cat > "${SNIP}" <<'EOF'
tracking:
  color: cnn_c2f
  cnn_c2f_version: B
  cnn_layer_fine: conv1
  cnn_channels_fine: 4
  cnn_channel_select_fine: "d5,d29,d40,d52"
  cnn_layer_full_channels_fine: 64
  cnn_layer_coarse: layer2
  cnn_channels_coarse: 6
  cnn_channel_select_coarse: "d7,d37,d60,d67,d74,d104"
  cnn_layer_full_channels_coarse: 128
  cnn_mode: cnn_only
EOF
        ;;

    # ------------------------------------------------------------------
    # C2F version B — FreqDerived channels
    # ------------------------------------------------------------------
    c2f_b_FreqDerived)
        cat > "${SNIP}" <<'EOF'
tracking:
  color: cnn_c2f
  cnn_c2f_version: B
  cnn_layer_fine: conv1
  cnn_channels_fine: 4
  cnn_channel_select_fine: "d5,d25,d22,d6"
  cnn_layer_full_channels_fine: 64
  cnn_layer_coarse: layer2
  cnn_channels_coarse: 4
  cnn_channel_select_coarse: "d121,d67,d60,d51"
  cnn_layer_full_channels_coarse: 128
  cnn_mode: cnn_only
EOF
        ;;

    # ------------------------------------------------------------------
    # C2F version B — PrincipleBased channels
    # ------------------------------------------------------------------
    c2f_b_PrincipleBased)
        cat > "${SNIP}" <<'EOF'
tracking:
  color: cnn_c2f
  cnn_c2f_version: B
  cnn_layer_fine: conv1
  cnn_channels_fine: 4
  cnn_channel_select_fine: "d5,d6,d52,d15"
  cnn_layer_full_channels_fine: 64
  cnn_layer_coarse: layer2
  cnn_channels_coarse: 4
  cnn_channel_select_coarse: "d51,d60,d67,d9"
  cnn_layer_full_channels_coarse: 128
  cnn_mode: cnn_only
EOF
        ;;

    *)
        echo "[ERROR] Unknown experiment: ${EXP}"
        return 1
        ;;
    esac
}

# ── Apply snippet to como.yml via Python ──────────────────────────────────────
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
print(f"[Config] Applied: {tag}  |  Experiment: ${EXP}")
PYEOF
}

# ── Trajectory validation via Python ──────────────────────────────────────────
# Returns exit code 0 = valid, 1 = NaN detected (prints reason to stdout)
check_traj() {
    local TRAJ="$1"

    python3 - <<PYEOF
import sys

traj_path = "${TRAJ}"

full_text = open(traj_path).read().lower()
if "nan" in full_text:
    print("NAN_FAIL")
    sys.exit(1)

print("OK")
sys.exit(0)
PYEOF
}

# ── Single run + ATE/RPE evaluation ───────────────────────────────────────────
run_one() {
    local EXP="$1"
    local DS_KEY="$2"
    local DS_PATH="${DATASET_ROOT}/${DATASETS[${DS_KEY}]}"
    local GT_FILE="${DS_PATH}/groundtruth.txt"
    local SAVE_NAME="${EXP}__${DS_KEY}"
    local TRAJ_SAVED="${RESULTS_DIR}/${SAVE_NAME}.txt"

    echo ""
    echo "##################################################"
    echo "# EXP: ${EXP}  |  DATASET: ${DS_KEY}"
    echo "##################################################"

    write_snippet "${EXP}"
    apply_snippet "${EXP}"

    # Remove stale trajectory
    rm -f "${TRAJ_FILE}"

    # Run COMO with timeout
    timeout 900 python como/como_dataset.py \
        --dataset_type=tum \
        --dataset_dir="${DS_PATH}" \
        || true

    # ── Failure case 1: trajectory file not produced ──────────────────────────
    if [ ! -f "${TRAJ_FILE}" ]; then
        echo "  [FAIL] Trajectory not produced — COMO crashed or timed out."
        echo -e "${DS_KEY}\t${EXP}\t-\t-\t-\tFAIL" >> "${SUMMARY_FILE}"
        return
    fi

    # ── Failure case 2: NaN check ─────────────────────────────────────────────────
    VALIDITY=$(check_traj "${TRAJ_FILE}")
    VCODE=$?

    if [ "${VCODE}" -ne 0 ]; then
        echo "  [FAIL] Trajectory invalid: ${VALIDITY}"
        # Save the bad trajectory with a suffix for post-hoc inspection
        cp "${TRAJ_FILE}" "${TRAJ_SAVED}.${VALIDITY%%_*}.bad"
        echo -e "${DS_KEY}\t${EXP}\t-\t-\t-\t${VALIDITY}" >> "${SUMMARY_FILE}"
        return
    fi

    # ── Valid trajectory: compute ATE + RPE ───────────────────────────────────
    cp "${TRAJ_FILE}" "${TRAJ_SAVED}"

    ATE_RMSE=$(evo_ape tum "${GT_FILE}" "${TRAJ_SAVED}" \
        --align --correct_scale 2>/dev/null \
        | grep "rmse" | awk '{printf "%.4f", $NF * 100}')
    ATE_MEAN=$(evo_ape tum "${GT_FILE}" "${TRAJ_SAVED}" \
        --align --correct_scale 2>/dev/null \
        | grep "mean" | awk '{printf "%.4f", $NF * 100}')
    RPE_RMSE=$(evo_rpe tum "${GT_FILE}" "${TRAJ_SAVED}" \
        --align --correct_scale 2>/dev/null \
        | grep "rmse" | awk '{printf "%.4f", $NF * 100}')

    if [ -z "${ATE_RMSE}" ]; then
        echo "  [FAIL] ATE parse error"
        echo -e "${DS_KEY}\t${EXP}\t-\t-\t-\tPARSE_FAIL" >> "${SUMMARY_FILE}"
        return
    fi

    echo "  ATE RMSE=${ATE_RMSE} cm  |  ATE Mean=${ATE_MEAN} cm  |  RPE RMSE=${RPE_RMSE} cm  |  OK"
    echo -e "${DS_KEY}\t${EXP}\t${ATE_RMSE}\t${ATE_MEAN}\t${RPE_RMSE}\tOK" >> "${SUMMARY_FILE}"
}

# ── Main ───────────────────────────────────────────────────────────────────────
cp "${CONFIG_FILE}" "${CONFIG_BACKUP}"
trap "echo '[Cleanup] Restoring original config...'; cp '${CONFIG_BACKUP}' '${CONFIG_FILE}'" EXIT

cd "${PROJECT_DIR}"

# Write TSV header
echo -e "Dataset\tExperiment\tATE_RMSE_cm\tATE_Mean_cm\tRPE_RMSE_cm\tStatus" > "${SUMMARY_FILE}"

echo ""
echo "Starting: ${#CONFIGS[@]} configs × ${#DATASET_KEYS[@]} sequences = $((${#CONFIGS[@]} * ${#DATASET_KEYS[@]})) runs"
echo ""

# Outer loop: datasets; inner loop: configs
for DS_KEY in "${DATASET_KEYS[@]}"; do
    for EXP in "${CONFIGS[@]}"; do
        run_one "${EXP}" "${DS_KEY}"
    done
done

# ── Final summary table ────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "FINAL SUMMARY"
echo "=========================================="

for DS_KEY in "${DATASET_KEYS[@]}"; do
    echo ""
    echo "  Dataset: ${DS_KEY}"
    printf "  %-26s  %14s  %14s  %14s  %10s\n" \
        "Experiment" "ATE RMSE(cm)" "ATE Mean(cm)" "RPE RMSE(cm)" "Status"
    printf "  %-26s  %14s  %14s  %14s  %10s\n" \
        "--------------------------" "--------------" "--------------" "--------------" "----------"
    grep "^${DS_KEY}" "${SUMMARY_FILE}" | while IFS=$'\t' read -r ds exp ate_r ate_m rpe_r stat; do
        printf "  %-26s  %14s  %14s  %14s  %10s\n" "${exp}" "${ate_r}" "${ate_m}" "${rpe_r}" "${stat}"
    done
done

# ── Failure summary ────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "FAILURE SUMMARY"
echo "=========================================="
FAIL_COUNT=$(grep -v "^Dataset" "${SUMMARY_FILE}" | grep -v $'\tOK$' | wc -l)
echo "  Total failures: ${FAIL_COUNT} / $((${#CONFIGS[@]} * ${#DATASET_KEYS[@]}))"
echo ""
grep -v "^Dataset" "${SUMMARY_FILE}" | grep -v $'\tOK$' | \
    while IFS=$'\t' read -r ds exp ate_r ate_m rpe_r stat; do
        printf "  %-20s  %-26s  %s\n" "${ds}" "${exp}" "${stat}"
    done

echo ""
echo "Full results: ${SUMMARY_FILE}"
echo "Trajectories: ${RESULTS_DIR}/"
echo "All done at $(date)"
echo "=========================================="
