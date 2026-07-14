#!/bin/bash
# ============================================================
# run_failure_threshold.sh
# Failure-threshold sweep: online lightswitch augmentation
# with intensity_scale, no extra RGB copies on disk.
#
# Strategy (from P3 design doc):
#   Phase 1 coarse scan: 0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50
#   Methods: gray, fine_only, c2f_b
#   Dataset: fr3/long_office_household (representative long sequence)
#
# The lightswitch effect is applied online in the dataset loader
# via a thin wrapper script (apply_lightswitch_online.py).
# depth, groundtruth, timestamps are untouched.
# ============================================================

set -euo pipefail

PROJECT_DIR="/home/melody/code/individual_project/como"
CONFIG_FILE="${PROJECT_DIR}/config/como.yml"
CONFIG_BACKUP="${PROJECT_DIR}/config/como.yml.bak"
TRAJ_FILE="${PROJECT_DIR}/results/data_tum.txt"
RESULTS_DIR="${PROJECT_DIR}/results/failure_threshold"
DATASET_DIR="/home/melody/data/tum/rgbd_dataset_freiburg3_long_office_household"
GT_FILE="${DATASET_DIR}/groundtruth.txt"
SYNTH_SCRIPT="${PROJECT_DIR}/como/lighting_synthesis_v2.py"

mkdir -p "${RESULTS_DIR}" "${PROJECT_DIR}/logs"

# ── Intensity scales to sweep (Phase 1 coarse) ──────────────────────────────
SCALES=("0.00" "0.25" "0.50" "0.75" "1.00" "1.25" "1.50")

# ── Methods ──────────────────────────────────────────────────────────────────
METHODS=("gray" "fine_only" "c2f_b")

# ── Failure criteria ─────────────────────────────────────────────────────────
# ATE Mean threshold: 2x the clean ATE for fr3 gray (11.35 cm) = 22.70 cm
# If ATE Mean > this, mark as ACCURACY_FAIL even if tracking completed.
CLEAN_ATE_CM="11.35"
ATE_FAIL_MULTIPLIER="2.0"
ATE_FAIL_THRESH=$(echo "${CLEAN_ATE_CM} * ${ATE_FAIL_MULTIPLIER}" | bc)

# ── Config snippets ──────────────────────────────────────────────────────────
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

# ── Online lightswitch augmentation wrapper ──────────────────────────────────
# Generates a temporary augmented dataset directory using symlinks for
# depth/groundtruth/timestamps, and a patched rgb.txt pointing to
# on-the-fly synthesized images written to /tmp/lightswitch_rgb/.
# scale=0.00 means no augmentation (clean).
apply_lightswitch_scale() {
    local SCALE="$1"
    local AUG_DIR="/tmp/como_aug_scale_${SCALE}"
    rm -rf "${AUG_DIR}"
    mkdir -p "${AUG_DIR}"

    # Symlink everything except rgb images
    for f in depth.txt groundtruth.txt intrinsics.txt matched_depth.txt matched_rgb.txt; do
        [ -f "${DATASET_DIR}/${f}" ] && ln -s "${DATASET_DIR}/${f}" "${AUG_DIR}/${f}"
    done
    ln -s "${DATASET_DIR}/depth" "${AUG_DIR}/depth"

    if python3 -c "import sys; sys.exit(0 if float('${SCALE}') < 0.001 else 1)" 2>/dev/null; then
        # scale=0: just symlink original rgb directly
        ln -s "${DATASET_DIR}/rgb" "${AUG_DIR}/rgb"
        ln -s "${DATASET_DIR}/rgb.txt" "${AUG_DIR}/rgb.txt"
        echo "[AugSetup] scale=0.00 → using original clean RGB"
    else
        # Generate augmented RGB frames into AUG_DIR/rgb/
        mkdir -p "${AUG_DIR}/rgb"
        echo "[AugSetup] Generating lightswitch RGB at scale=${SCALE} ..."
        python3 "${PROJECT_DIR}/como/apply_lightswitch_online.py" \
            --dataset-dir "${DATASET_DIR}" \
            --out-dir "${AUG_DIR}" \
            --intensity-scale "${SCALE}" \
            --seed 42
        echo "[AugSetup] Done."
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

    echo ""
    echo "##################################################"
    echo "# METHOD: ${METHOD}  |  SCALE: ${SCALE}"
    echo "##################################################"

    write_snippet "${METHOD}"
    apply_snippet "${METHOD}"
    rm -f "${TRAJ_FILE}"

    timeout 300 python como/como_dataset.py \
        --dataset_type=tum \
        --dataset_dir="${AUG_DIR}" || true

    # ── Check trajectory produced ────────────────────────────────────────────
    if [ ! -f "${TRAJ_FILE}" ]; then
        STATUS="TRACKING_FAIL"
        ATE_MEAN="-"
        echo "  [FAIL] No trajectory produced."
        echo -e "${METHOD}\t${SCALE}\t${ATE_MEAN}\t${STATUS}" >> "${SUMMARY_FILE}"
        return
    fi

    # ── NaN guard ────────────────────────────────────────────────────────────
    if grep -qi "nan" "${TRAJ_FILE}"; then
        STATUS="NAN_FAIL"
        ATE_MEAN="-"
        echo "  [FAIL] Trajectory contains NaN."
        cp "${TRAJ_FILE}" "${TRAJ_SAVED}.nan"
        echo -e "${METHOD}\t${SCALE}\t${ATE_MEAN}\t${STATUS}" >> "${SUMMARY_FILE}"
        return
    fi

    cp "${TRAJ_FILE}" "${TRAJ_SAVED}"

    # ── ATE Mean ─────────────────────────────────────────────────────────────
    ATE_MEAN=$(evo_ape tum "${GT_FILE}" "${TRAJ_SAVED}" \
        --align --correct_scale 2>/dev/null \
        | grep -w "mean" | awk '{printf "%.4f", $2*100}')

    if [ -z "${ATE_MEAN}" ]; then
        STATUS="PARSE_FAIL"
        ATE_MEAN="-"
    else
        # Check accuracy threshold: ATE > 2x clean ATE?
        OVER_THRESH=$(python3 -c "print('1' if float('${ATE_MEAN}') > ${ATE_FAIL_THRESH} else '0')" 2>/dev/null || echo "0")
        if [ "${OVER_THRESH}" = "1" ]; then
            STATUS="ACCURACY_FAIL"
        else
            STATUS="OK"
        fi
    fi

    echo "  ATE Mean = ${ATE_MEAN} cm  |  ${STATUS}  (threshold=${ATE_FAIL_THRESH} cm)"
    echo -e "${METHOD}\t${SCALE}\t${ATE_MEAN}\t${STATUS}" >> "${SUMMARY_FILE}"
}

# ── Main ──────────────────────────────────────────────────────────────────────
cp "${CONFIG_FILE}" "${CONFIG_BACKUP}"
trap "echo '[Cleanup] Restoring config...'; cp '${CONFIG_BACKUP}' '${CONFIG_FILE}'" EXIT

cd "${PROJECT_DIR}"

SUMMARY_FILE="${RESULTS_DIR}/summary.tsv"
echo -e "Method\tIntensity_Scale\tATE_Mean_cm\tStatus" > "${SUMMARY_FILE}"

echo "=========================================="
echo "Failure Threshold Sweep — Lightswitch"
echo "Dataset: fr3/long_office_household"
echo "Scales: ${SCALES[*]}"
echo "Methods: ${METHODS[*]}"
echo "ATE accuracy-fail threshold: ${ATE_FAIL_THRESH} cm (${ATE_FAIL_MULTIPLIER}x clean)"
echo "=========================================="

for SCALE in "${SCALES[@]}"; do
    echo ""
    echo "========== Preparing augmented dataset at scale=${SCALE} =========="
    AUG_DIR=$(apply_lightswitch_scale "${SCALE}")

    for METHOD in "${METHODS[@]}"; do
        run_one "${METHOD}" "${SCALE}" "${AUG_DIR}"
    done

    # Clean up augmented RGB to save disk (keep scale=0 clean symlink)
    if python3 -c "import sys; sys.exit(0 if float('${SCALE}') < 0.001 else 1)" 2>/dev/null; then
        :
    else
        rm -rf "${AUG_DIR}/rgb"
    fi
done

# ── Final summary ─────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "FINAL SUMMARY — Failure Threshold Sweep"
echo "=========================================="
printf "%-14s  %-8s  %-14s  %-16s\n" "Method" "Scale" "ATE Mean (cm)" "Status"
echo "------------------------------------------------------------"
tail -n +2 "${SUMMARY_FILE}" | while IFS=$'\t' read -r method scale ate stat; do
    printf "%-14s  %-8s  %-14s  %-16s\n" "${method}" "${scale}" "${ate}" "${stat}"
done
echo ""
echo "All done at $(date)"
echo "Results saved to: ${SUMMARY_FILE}"