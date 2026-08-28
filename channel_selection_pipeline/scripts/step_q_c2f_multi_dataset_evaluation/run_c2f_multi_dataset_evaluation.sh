#!/usr/bin/env bash
# Run/resume the focused parent-versus-C2F comparison over 3 TUM families × 3
# lighting conditions.  The two architectures are intentionally serial because
# COMO uses one shared configuration file.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
PYTHON="/home/melody/anaconda3/envs/como/bin/python"
EVALUATOR="$SCRIPT_DIR/run_c2f_parent_comparison.py"
AGGREGATOR="$SCRIPT_DIR/aggregate_c2f_parent_comparison.py"
DATASET_PLAN="$SCRIPT_DIR/c2f_multi_dataset_plan.json"
RESULTS_ROOT="$PROJECT_ROOT/channel_selection_results/step_q_c2f_multi_dataset_evaluation"
COMO_DIR="$PROJECT_ROOT/como"
CONFIG="$COMO_DIR/config/como.yml"
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
  run_c2f_multi_dataset_evaluation.sh unet                 # validate only
  run_c2f_multi_dataset_evaluation.sh unet --execute       # run/resume 90 U-Net cells
  run_c2f_multi_dataset_evaluation.sh resnet               # validate only
  run_c2f_multi_dataset_evaluation.sh resnet --execute     # run/resume 90 ResNet cells
  run_c2f_multi_dataset_evaluation.sh <architecture> --aggregate-only

Each dataset has its own SQLite database.  A rerun skips every saved
configuration (PASS and saved failures alike), so an interruption resumes at
the first missing row.  Run U-Net and ResNet serially: both use the same COMO
configuration file, although each architecture has separate result stores.
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
    case "$2" in
        --execute) MODE="execute" ;;
        --aggregate-only) MODE="aggregate" ;;
        *) usage >&2; exit 2 ;;
    esac
fi

CANDIDATE_PLAN="$SCRIPT_DIR/${ARCHITECTURE}_c2f_parent_comparison_plan.json"
OUTPUT_DIR="$RESULTS_ROOT/$ARCHITECTURE"
for required in "$PYTHON" "$EVALUATOR" "$AGGREGATOR" "$DATASET_PLAN" "$CANDIDATE_PLAN" "$CONFIG"; do
    [[ -e "$required" ]] || die "Required path is missing: $required"
done

mapfile -t DATASET_ROWS < <(
    "$PYTHON" - "$DATASET_PLAN" "$CANDIDATE_PLAN" "$ARCHITECTURE" <<'PY'
import json
import pathlib
import sys

dataset_path, candidate_path, architecture = sys.argv[1:]
datasets_doc = json.loads(pathlib.Path(dataset_path).read_text(encoding="utf-8"))
candidates_doc = json.loads(pathlib.Path(candidate_path).read_text(encoding="utf-8"))
if datasets_doc.get("protocol") != "c2f_parent_comparative_three_dataset_three_lighting_conditions_v1":
    raise SystemExit("[ABORT] Unexpected C2F multi-dataset protocol")
if candidates_doc.get("protocol") != "c2f_parent_comparative_candidate_plan_v1":
    raise SystemExit("[ABORT] Unexpected C2F candidate protocol")
if candidates_doc.get("architecture") != architecture:
    raise SystemExit("[ABORT] Candidate plan architecture does not match launcher argument")
datasets = datasets_doc.get("datasets", [])
candidates = candidates_doc.get("candidates", [])
if len(datasets) != 9 or datasets_doc.get("dataset_count") != 9:
    raise SystemExit("[ABORT] Expected exactly nine datasets")
if len(candidates) != candidates_doc.get("selection", {}).get("selected_count"):
    raise SystemExit("[ABORT] Candidate selection count does not match plan")
if datasets_doc.get("timeout_seconds_per_run") != 500 or datasets_doc.get("replicates_per_candidate") != 1:
    raise SystemExit("[ABORT] Expected one replicate and a 500-second timeout")
if len({item.get("label") for item in candidates}) != len(candidates):
    raise SystemExit("[ABORT] Candidate labels are not unique")
root = pathlib.Path(datasets_doc["dataset_root"])
for order, item in enumerate(datasets, start=1):
    if item.get("order") != order:
        raise SystemExit("[ABORT] Dataset order is not frozen")
    dataset = root / item["directory_name"]
    for name in ("matched_rgb.txt", "matched_depth.txt", "groundtruth.txt"):
        if not (dataset / name).is_file():
            raise SystemExit(f"[ABORT] Missing {dataset / name}")
    with (dataset / "matched_rgb.txt").open(encoding="utf-8") as handle:
        frames = sum(1 for line in handle if line.strip() and not line.lstrip().startswith("#"))
    if frames != item["expected_matched_frames"]:
        raise SystemExit(f"[ABORT] {item['key']} frame count changed: {frames} != {item['expected_matched_frames']}")
    print("\t".join((item["key"], str(dataset), str(frames), item["family"], item["condition"])))
PY
)

(( ${#DATASET_ROWS[@]} == 9 )) || die "Dataset-plan validation returned ${#DATASET_ROWS[@]} rows"
candidate_count="$("$PYTHON" - "$CANDIDATE_PLAN" <<'PY'
import json
import pathlib
import sys
print(len(json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))["candidates"]))
PY
)"

printf '%s\n' '=============================================================================='
printf '%s C2F PARENT-COMPARISON: THREE DATASET FAMILIES × THREE LIGHTING CONDITIONS\n' "${ARCHITECTURE^^}"
printf '%s\n' '=============================================================================='
printf '[MODE] %s\n' "$MODE"
printf '[PLAN] 9 sequences × %s focused configurations = %s nominal cells\n' "$candidate_count" "$((9 * candidate_count))"
printf '[DESIGN] Direct parents and C2F configurations are evaluated together; aggregation reports within-dataset parent deltas\n'
printf '[MAPPING] gray + sensor depth; tracking receives direct or C2F learned features only\n'
printf '[METRIC] historical keyframe evo_ape ATE mean (--align --correct_scale); full diagnostics retained\n'
printf '[TIMEOUT] %ds per run\n' "$TIMEOUT_SECONDS"
printf '[OUTPUT] %s\n' "$OUTPUT_DIR"
for row in "${DATASET_ROWS[@]}"; do
    IFS=$'\t' read -r key path frames family condition <<< "$row"
    printf '  [%s] frames=%s family=%s condition=%s\n' "$key" "$frames" "$family" "$condition"
done

if [[ "$MODE" == "aggregate" ]]; then
    "$PYTHON" "$AGGREGATOR" --architecture "$ARCHITECTURE" --output-dir "$OUTPUT_DIR" \
        --dataset-plan "$DATASET_PLAN" --candidate-plan "$CANDIDATE_PLAN"
    exit 0
fi

if [[ "$MODE" == "validate" ]]; then
    for row in "${DATASET_ROWS[@]}"; do
        IFS=$'\t' read -r key path _frames _family _condition <<< "$row"
        "$PYTHON" "$EVALUATOR" --architecture "$ARCHITECTURE" --dataset-dir "$path" \
            --candidate-plan "$CANDIDATE_PLAN" --timeout-seconds "$TIMEOUT_SECONDS"
        printf '[VALID] %s\n' "$key"
    done
    printf '\n[VALIDATION COMPLETE] No COMO process, SQLite row, trajectory, or config edit was created.\n'
    printf '[RUN] %s %s --execute\n' "$0" "$ARCHITECTURE"
    exit 0
fi

command -v nvidia-smi >/dev/null 2>&1 || die 'nvidia-smi is unavailable'
command -v taskset >/dev/null 2>&1 || die 'taskset is unavailable'
command -v flock >/dev/null 2>&1 || die 'flock is unavailable'
taskset -c "$CPU_AFFINITY" true 2>/dev/null || die "Invalid CPU affinity: $CPU_AFFINITY"
[[ -r "$NO_TURBO_PATH" ]] || die 'Cannot verify Intel Turbo state'
[[ "$(<"$NO_TURBO_PATH")" == "1" ]] || die "Intel Turbo is enabled; run: echo 1 | sudo tee $NO_TURBO_PATH"
printf '[SAFETY] GPU: '
nvidia-smi -L || die 'NVIDIA driver/GPU is unavailable'
printf '[SAFETY] CPU affinity=%s; Intel Turbo=disabled\n' "$CPU_AFFINITY"

mkdir -p "$OUTPUT_DIR/launch_backups" "$RESULTS_ROOT"
exec 9>"$RESULTS_ROOT/.c2f_multi_dataset_global.lock"
flock -n 9 || die 'Another C2F multi-dataset launcher is already active; do not run U-Net and ResNet concurrently.'
stamp="$(date '+%Y%m%d_%H%M%S')"
backup_dir="$OUTPUT_DIR/launch_backups/$stamp"
mkdir -p "$backup_dir/databases"
cp -- "$CONFIG" "$backup_dir/como_before_launch.yml"
cp -- "$DATASET_PLAN" "$backup_dir/c2f_multi_dataset_plan.json"
cp -- "$CANDIDATE_PLAN" "$backup_dir/${ARCHITECTURE}_c2f_parent_comparison_plan.json"

restore_config() {
    if [[ -f "$backup_dir/como_before_launch.yml" ]]; then
        cp -- "$backup_dir/como_before_launch.yml" "$CONFIG"
        printf '[CLEANUP] Restored the pre-launch COMO configuration.\n'
    fi
}
trap restore_config EXIT

for row in "${DATASET_ROWS[@]}"; do
    IFS=$'\t' read -r key _path _frames _family _condition <<< "$row"
    database="$OUTPUT_DIR/per_dataset/$key/evaluations.sqlite3"
    if [[ -f "$database" ]]; then
        "$PYTHON" - "$database" "$backup_dir/databases/${key}.sqlite3" <<'PY'
import sqlite3
import sys

source_path, backup_path = sys.argv[1:]
source = sqlite3.connect(f"file:{source_path}?mode=ro", uri=True)
if source.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
    raise SystemExit(f"[ABORT] SQLite integrity check failed: {source_path}")
target = sqlite3.connect(backup_path)
source.backup(target)
target.close()
source.close()
print(f"[BACKUP] SQLite snapshot: {backup_path}")
PY
    fi
done
printf '[BACKUP] Launch snapshot: %s\n' "$backup_dir"

for row in "${DATASET_ROWS[@]}"; do
    IFS=$'\t' read -r key path frames family condition <<< "$row"
    dataset_output="$OUTPUT_DIR/per_dataset/$key"
    printf '\n[DATASET START] %s (%s frames; %s/%s)\n' "$key" "$frames" "$family" "$condition"
    taskset -c "$CPU_AFFINITY" "$PYTHON" "$EVALUATOR" \
        --architecture "$ARCHITECTURE" --execute --dataset-dir "$path" --output-dir "$dataset_output" \
        --candidate-plan "$CANDIDATE_PLAN" --timeout-seconds "$TIMEOUT_SECONDS"
    "$PYTHON" "$AGGREGATOR" --architecture "$ARCHITECTURE" --output-dir "$OUTPUT_DIR" \
        --dataset-plan "$DATASET_PLAN" --candidate-plan "$CANDIDATE_PLAN"
done

"$PYTHON" "$AGGREGATOR" --architecture "$ARCHITECTURE" --output-dir "$OUTPUT_DIR" \
    --dataset-plan "$DATASET_PLAN" --candidate-plan "$CANDIDATE_PLAN"
printf '\n[DONE] %s parent-versus-C2F evaluation completed/resumed across %s cells.\n' "$ARCHITECTURE" "$((9 * candidate_count))"
printf '[READ FIRST] %s/aggregate_summary.md\n' "$OUTPUT_DIR"
printf '[EVIDENCE] %s/c2f_pairwise_comparison.csv\n' "$OUTPUT_DIR"
printf '[PRESENTATION TABLE] %s/ate_mean_matrix.csv\n' "$OUTPUT_DIR"
