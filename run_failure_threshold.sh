#!/bin/bash
# ============================================================
# run_failure_threshold.sh
# Failure-threshold sweep: online lightswitch augmentation
# with intensity_scale.
#
# Fine scan: 0.00 to 1.50 in steps of 0.10 (16 levels)
#
# Methods (7 configs from Step 4 Fr1/desk table):
#   gray, fine_only, coarse_only,
#   c2f_a_BestATE, c2f_a_FreqDerived,
#   c2f_b_BestATE, c2f_b_FreqDerived
#
# Dataset: fr1/desk
#
# Failure criteria:
#   TRACKING_FAIL  — timeout (>200s) or no trajectory file produced
#   NAN_FAIL       — stdout contains "[KF aff received]...nan" OR
#                    trajectory file contains "nan"
#
# Total runs: 7 methods × 7 scales = 49
# ============================================================

set -uo pipefail

PROJECT_DIR="/home/melody/code/individual_project/como"
CONFIG_FILE="${PROJECT_DIR}/config/como.yml"
CONFIG_BACKUP="${PROJECT_DIR}/config/como.yml.bak"
# TRAJ_FILE is computed dynamically inside run_one based on AUG_DIR basename
RESULTS_DIR="${PROJECT_DIR}/results/failure_threshold"
DATASET_DIR="/home/melody/data/tum/rgbd_dataset_freiburg1_desk"
GT_FILE="${DATASET_DIR}/groundtruth.txt"
APPLY_SCRIPT_DIR="/home/melody/code/individual_project"
APPLY_SCRIPT="${APPLY_SCRIPT_DIR}/apply_lightswitch_online.py"

mkdir -p "${RESULTS_DIR}" "${PROJECT_DIR}/logs"

# ── Always run from PROJECT_DIR so python como/como_dataset.py resolves correctly
cp "${CONFIG_FILE}" "${CONFIG_BACKUP}"
trap "echo '[Cleanup] Restoring config...'; cp '${CONFIG_BACKUP}' '${CONFIG_FILE}'" EXIT
cd "${PROJECT_DIR}"

# ── Intensity scales to sweep (Phase 1 coarse) ──────────────────────────────
SCALES=("0.00" "0.10" "0.20" "0.30" "0.40" "0.50" "0.60" "0.70" "0.80" "0.90" "1.00" "1.10" "1.20" "1.30" "1.40" "1.50")

# ── Methods ──────────────────────────────────────────────────────────────────
METHODS=(
    "gray"
    "fine_only"
    "coarse_only"
    "c2f_a_BestATE"
    "c2f_a_FreqDerived"
    "c2f_b_BestATE"
    "c2f_b_FreqDerived"
)

# ── Config snippets ──────────────────────────────────────────────────────────
write_snippet() {
    local EXP="$1"
    local SNIP="/tmp/snip_${EXP}.yml"
    case "${EXP}" in

    gray)
        cat > "${SNIP}" <<'EOF'
tracking:
  color: gray
EOF
        ;;

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

    *)
        echo "[ERROR] Unknown experiment: ${EXP}"
        return 1
        ;;
    esac
}

apply_snippet() {
    local EXP="$1"
    local SNIP="/tmp/snip_${EXP}.yml"
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

# ── Prepare augmented dataset for a given scale ──────────────────────────────
prepare_aug_dataset() {
    local SCALE="$1"
    local AUG_DIR="/tmp/como_aug_scale_${SCALE}"
    rm -rf "${AUG_DIR}"
    mkdir -p "${AUG_DIR}"

    # Symlink depth, groundtruth, calibration files
    for f in depth.txt groundtruth.txt intrinsics.txt matched_depth.txt matched_rgb.txt calibration.txt; do
        [ -f "${DATASET_DIR}/${f}" ] && ln -s "${DATASET_DIR}/${f}" "${AUG_DIR}/${f}"
    done
    [ -d "${DATASET_DIR}/depth" ] && ln -s "${DATASET_DIR}/depth" "${AUG_DIR}/depth"

    if python3 -c "import sys; sys.exit(0 if float('${SCALE}') < 0.001 else 1)" 2>/dev/null; then
        # scale=0: symlink original clean RGB
        ln -s "${DATASET_DIR}/rgb"     "${AUG_DIR}/rgb"
        ln -s "${DATASET_DIR}/rgb.txt" "${AUG_DIR}/rgb.txt"
        echo "[AugSetup] scale=0.00 → using original clean RGB" >&2
    else
        mkdir -p "${AUG_DIR}/rgb"
        echo "[AugSetup] Generating lightswitch RGB at scale=${SCALE} ..." >&2
        (cd "${APPLY_SCRIPT_DIR}" && python3 "${APPLY_SCRIPT}" \
            --dataset-dir "${DATASET_DIR}" \
            --out-dir     "${AUG_DIR}" \
            --intensity-scale "${SCALE}" \
            --seed 42)
        echo "[AugSetup] Done." >&2
    fi

    echo "${AUG_DIR}"
}

# ── Run one experiment ────────────────────────────────────────────────────────
run_one() {
    local METHOD="$1"
    local SCALE="$2"
    local AUG_DIR="$3"
    local SUMMARY_FILE="${RESULTS_DIR}/summary.tsv"
    local TAG="${METHOD}__scale_${SCALE}"
    local TRAJ_SAVED="${RESULTS_DIR}/${TAG}.txt"
    local STDOUT_LOG="${RESULTS_DIR}/${TAG}.log"

    echo ""
    echo "##################################################"
    echo "# METHOD: ${METHOD}  |  SCALE: ${SCALE}"
    echo "##################################################"

    # COMO names traj as parts[1]+"_"+parts[2] of AUG_DIR split by "/"
    local _parts
    IFS="/" read -ra _parts <<< "${AUG_DIR}"
    local TRAJ_FILE="${PROJECT_DIR}/results/${_parts[1]}_${_parts[2]}.txt"

    write_snippet "${METHOD}"
    apply_snippet "${METHOD}"
    rm -f "${TRAJ_FILE}"

    # Run COMO; capture stdout+stderr to log file for NaN detection
    timeout 200 python como/como_dataset.py \
        --dataset_type=tum \
        --dataset_dir="${AUG_DIR}" \
        2>&1 | tee "${STDOUT_LOG}" || true

    # ── Failure case 1: no trajectory produced ───────────────────────────────
    if [ ! -f "${TRAJ_FILE}" ]; then
        echo "  [FAIL] No trajectory produced (crash or timeout)."
        echo -e "${METHOD}\t${SCALE}\t-\tTRACKING_FAIL" >> "${SUMMARY_FILE}"
        return
    fi

    # ── Failure case 2: NaN in stdout (kf_aff nan) or trajectory file ────────
    if grep -qi "nan" "${STDOUT_LOG}" || grep -qi "nan" "${TRAJ_FILE}"; then
        echo "  [FAIL] NaN detected in stdout or trajectory."
        cp "${TRAJ_FILE}" "${TRAJ_SAVED}.nan.bad"
        echo -e "${METHOD}\t${SCALE}\t-\tNAN_FAIL" >> "${SUMMARY_FILE}"
        return
    fi

    cp "${TRAJ_FILE}" "${TRAJ_SAVED}"

    # ── Compute ATE Mean ─────────────────────────────────────────────────────
    ATE_MEAN=$(evo_ape tum "${GT_FILE}" "${TRAJ_SAVED}" \
        --align --correct_scale 2>/dev/null \
        | grep "mean" | awk '{printf "%.4f", $NF * 100}')

    if [ -z "${ATE_MEAN}" ]; then
        echo "  [FAIL] ATE parse error."
        echo -e "${METHOD}\t${SCALE}\t-\tPARSE_FAIL" >> "${SUMMARY_FILE}"
        return
    fi

    echo "  ATE Mean = ${ATE_MEAN} cm  |  OK"
    echo -e "${METHOD}\t${SCALE}\t${ATE_MEAN}\tOK" >> "${SUMMARY_FILE}"
}

# ── Main ──────────────────────────────────────────────────────────────────────
cp "${CONFIG_FILE}" "${CONFIG_BACKUP}"
trap "echo '[Cleanup] Restoring config...'; cp '${CONFIG_BACKUP}' '${CONFIG_FILE}'" EXIT

cd "${PROJECT_DIR}"

SUMMARY_FILE="${RESULTS_DIR}/summary.tsv"
echo -e "Method\tIntensity_Scale\tATE_Mean_cm\tStatus" > "${SUMMARY_FILE}"

echo "=========================================="
echo "Failure Threshold Sweep — fr1/desk"
echo "Scales: ${SCALES[*]}"
echo "Methods: ${METHODS[*]}"
echo "Total runs: $((${#METHODS[@]} * ${#SCALES[@]}))"
echo "Timeout per run: 200s"
echo "=========================================="

for SCALE in "${SCALES[@]}"; do
    echo ""
    echo "========== Preparing augmented dataset at scale=${SCALE} =========="
    AUG_DIR=$(prepare_aug_dataset "${SCALE}")

    for METHOD in "${METHODS[@]}"; do
        run_one "${METHOD}" "${SCALE}" "${AUG_DIR}"
    done

    # Clean up augmented RGB to save disk (keep scale=0 symlink)
    if python3 -c "import sys; sys.exit(0 if float('${SCALE}') < 0.001 else 1)" 2>/dev/null; then
        :
    else
        rm -rf "${AUG_DIR}/rgb"
    fi
done

# ── Final summary ─────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "FINAL SUMMARY — Failure Threshold Sweep (fr1/desk)"
echo "=========================================="
printf "%-22s  %-8s  %-14s  %-16s\n" "Method" "Scale" "ATE Mean (cm)" "Status"
echo "--------------------------------------------------------------------"
tail -n +2 "${SUMMARY_FILE}" | while IFS=$'\t' read -r method scale ate stat; do
    printf "%-22s  %-8s  %-14s  %-16s\n" "${method}" "${scale}" "${ate}" "${stat}"
done
echo ""
echo "All done at $(date)"
echo "Results saved to: ${SUMMARY_FILE}"
