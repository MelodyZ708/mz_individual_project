#!/usr/bin/env bash
# Run/resume one architecture's 72-cell C2F best-channel grid.
# Usage is intentionally architecture-separated because ResNet and U-Net share
# the same COMO configuration file and GPU but have independent SQLite stores.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
PYTHON="/home/melody/anaconda3/envs/como/bin/python"
EVALUATOR="$SCRIPT_DIR/run_c2f_best_channel_grid.py"
COMO_DIR="$PROJECT_ROOT/como"
CONFIG="$COMO_DIR/config/como.yml"
DATASET="/home/melody/data/tum/rgbd_dataset_freiburg1_desk_lightswitch"
RESULTS_ROOT="$PROJECT_ROOT/channel_selection_results/step_p_c2f_best_channels_evaluation"
CPU_AFFINITY="${CPU_AFFINITY:-0,1,4-9,12-15}"
NO_TURBO_PATH="/sys/devices/system/cpu/intel_pstate/no_turbo"
TIMEOUT_SECONDS=500

die() {
    printf '[ABORT] %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage:
  run_c2f_best_channel_grid.sh resnet                 # validation only
  run_c2f_best_channel_grid.sh resnet --execute       # run/resume 72 ResNet cells
  run_c2f_best_channel_grid.sh unet                   # validation only
  run_c2f_best_channel_grid.sh unet --execute         # run/resume 72 U-Net cells

The runs are architecture-separated and resumable.  Each architecture has one
SQLite database: an existing label/replicate record (PASS or failure) is
skipped automatically.  Do not run ResNet and U-Net launchers concurrently.
EOF
}

(( $# == 1 || $# == 2 )) || { usage >&2; exit 2; }
ARCHITECTURE="$1"
case "$ARCHITECTURE" in
    resnet|unet) ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

MODE="validate"
if (( $# == 2 )); then
    [[ "$2" == "--execute" ]] || { usage >&2; exit 2; }
    MODE="execute"
fi

PLAN="$SCRIPT_DIR/${ARCHITECTURE}_c2f_candidate_plan.json"
OUTPUT_DIR="$RESULTS_ROOT/${ARCHITECTURE}_fr1_desk_lightswitch"
for required in "$PYTHON" "$EVALUATOR" "$PLAN" "$CONFIG" \
    "$DATASET/matched_rgb.txt" "$DATASET/matched_depth.txt" "$DATASET/groundtruth.txt"; do
    [[ -e "$required" ]] || die "Required path is missing: $required"
done

printf '%s\n' '=============================================================================='
printf '[MODE] %s\n' "$MODE"
printf '[ARCHITECTURE] %s\n' "$ARCHITECTURE"
printf '[DATASET] %s\n' "$DATASET"
printf '[PLAN] 2 valid C2F variants (A/B) × 6 fine subsets × 6 coarse subsets = 72 runs\n'
printf '[MAPPING] gray + sensor depth; tracking receives C2F features only\n'
printf '[METRIC] keyframe evo_ape ATE mean (--align --correct_scale); diagnostics retained\n'
printf '[TIMEOUT] %ds per run\n' "$TIMEOUT_SECONDS"
printf '[OUTPUT] %s\n' "$OUTPUT_DIR"
printf '%s\n' '=============================================================================='

if [[ "$MODE" == "validate" ]]; then
    "$PYTHON" "$EVALUATOR" \
        --architecture "$ARCHITECTURE" \
        --dataset-dir "$DATASET" \
        --candidate-plan "$PLAN" \
        --timeout-seconds "$TIMEOUT_SECONDS"
    printf '\n[VALIDATION COMPLETE] No COMO process, SQLite row, trajectory, or config edit was created.\n'
    printf '[RUN] %s %s --execute\n' "$0" "$ARCHITECTURE"
    exit 0
fi

command -v nvidia-smi >/dev/null 2>&1 || die 'nvidia-smi is unavailable'
command -v taskset >/dev/null 2>&1 || die 'taskset is unavailable'
taskset -c "$CPU_AFFINITY" true 2>/dev/null || die "Invalid CPU affinity: $CPU_AFFINITY"
[[ -r "$NO_TURBO_PATH" ]] || die "Cannot verify Intel Turbo state"
[[ "$(<"$NO_TURBO_PATH")" == "1" ]] || \
    die "Intel Turbo is enabled; run: echo 1 | sudo tee $NO_TURBO_PATH"
printf '[SAFETY] GPU: '
nvidia-smi -L || die 'NVIDIA driver/GPU is unavailable'
printf '[SAFETY] CPU affinity=%s; Intel Turbo=disabled\n' "$CPU_AFFINITY"

mkdir -p "$OUTPUT_DIR/launch_backups"
stamp="$(date '+%Y%m%d_%H%M%S')"
backup_dir="$OUTPUT_DIR/launch_backups/$stamp"
mkdir -p "$backup_dir"
cp -- "$CONFIG" "$backup_dir/como_before_launch.yml"
cp -- "$PLAN" "$backup_dir/${ARCHITECTURE}_c2f_candidate_plan.json"

restore_config() {
    if [[ -f "$backup_dir/como_before_launch.yml" ]]; then
        cp -- "$backup_dir/como_before_launch.yml" "$CONFIG"
        printf '[CLEANUP] Restored the pre-launch COMO configuration.\n'
    fi
}
trap restore_config EXIT

if [[ -f "$OUTPUT_DIR/evaluations.sqlite3" ]]; then
    cp -- "$OUTPUT_DIR/evaluations.sqlite3" "$backup_dir/evaluations_before_launch.sqlite3"
    printf '[BACKUP] SQLite snapshot: %s\n' "$backup_dir/evaluations_before_launch.sqlite3"
fi
printf '[BACKUP] Launch snapshot: %s\n' "$backup_dir"

taskset -c "$CPU_AFFINITY" "$PYTHON" "$EVALUATOR" \
    --architecture "$ARCHITECTURE" \
    --execute \
    --dataset-dir "$DATASET" \
    --output-dir "$OUTPUT_DIR" \
    --candidate-plan "$PLAN" \
    --timeout-seconds "$TIMEOUT_SECONDS"

printf '\n[DONE] %s C2F grid completed or reached its current resumable state.\n' "$ARCHITECTURE"
printf '[READ FIRST] %s/summary.md\n' "$OUTPUT_DIR"
printf '[RANKING] %s/pass_ranking.csv\n' "$OUTPUT_DIR"
