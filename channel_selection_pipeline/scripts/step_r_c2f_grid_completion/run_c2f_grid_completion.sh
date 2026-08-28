#!/usr/bin/env bash
# Run/resume only cells missing from the selected complete C2F grids.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
PYTHON="/home/melody/anaconda3/envs/como/bin/python"
EVALUATOR="$SCRIPT_DIR/run_c2f_grid_completion.py"
AGGREGATOR="$SCRIPT_DIR/aggregate_c2f_grid_completion.py"
DATASET_PLAN="$PROJECT_ROOT/channel_selection_pipeline/scripts/step_q_c2f_multi_dataset_evaluation/c2f_multi_dataset_plan.json"
RESULTS_ROOT="$PROJECT_ROOT/channel_selection_results/step_r_c2f_grid_completion"
COMO_DIR="$PROJECT_ROOT/como"
CONFIG="$COMO_DIR/config/como.yml"
CPU_AFFINITY="${CPU_AFFINITY:-0,1,4-9,12-15}"
NO_TURBO_PATH="/sys/devices/system/cpu/intel_pstate/no_turbo"
TIMEOUT_SECONDS=500

die() { printf '[ABORT] %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'EOF'
Usage:
  run_c2f_grid_completion.sh unet                 # validate only; no Como run
  run_c2f_grid_completion.sh unet --execute       # run/resume 360 missing U-Net cells
  run_c2f_grid_completion.sh resnet               # validate only; no Como run
  run_c2f_grid_completion.sh resnet --execute     # run/resume 558 missing ResNet cells
  run_c2f_grid_completion.sh <architecture> --aggregate-only

Step-Q rows are checked and reused read-only. They are never copied or rerun.
Every new Step-R database is resumable: saved PASS and saved failure rows are
skipped on the next invocation. Run architectures serially because COMO has a
shared configuration file.
EOF
}

(( $# == 1 || $# == 2 )) || { usage >&2; exit 2; }
ARCHITECTURE="$1"
case "$ARCHITECTURE" in resnet|unet) ;; --help|-h) usage; exit 0 ;; *) usage >&2; exit 2;; esac
MODE="validate"
if (( $# == 2 )); then
    case "$2" in --execute) MODE="execute";; --aggregate-only) MODE="aggregate";; *) usage >&2; exit 2;; esac
fi

COMPLETION_PLAN="$SCRIPT_DIR/${ARCHITECTURE}_c2f_grid_completion_plan.json"
OUTPUT_DIR="$RESULTS_ROOT/$ARCHITECTURE"
for required in "$PYTHON" "$EVALUATOR" "$AGGREGATOR" "$COMPLETION_PLAN" "$DATASET_PLAN" "$CONFIG"; do
    [[ -e "$required" ]] || die "Required path is missing: $required"
done

mapfile -t DATASET_ROWS < <(
    "$PYTHON" - "$DATASET_PLAN" "$COMPLETION_PLAN" <<'PY'
import json
import pathlib
import sys

dataset_path, completion_path = map(pathlib.Path, sys.argv[1:])
datasets_doc = json.loads(dataset_path.read_text(encoding="utf-8"))
completion = json.loads(completion_path.read_text(encoding="utf-8"))
if datasets_doc.get("protocol") != "c2f_parent_comparative_three_dataset_three_lighting_conditions_v1":
    raise SystemExit("[ABORT] Unexpected nine-dataset protocol")
if completion.get("protocol") != "c2f_full_grid_completion_v1":
    raise SystemExit("[ABORT] Unexpected Step-R completion protocol")
datasets = datasets_doc.get("datasets", [])
expected = completion.get("expected", {})
if len(datasets) != 9 or datasets_doc.get("dataset_count") != 9:
    raise SystemExit("[ABORT] Expected nine datasets")
if completion.get("variants") != ["A", "B"] or completion.get("timeout_seconds") != 500:
    raise SystemExit("[ABORT] Expected A/B variants and 500-second timeout")
if expected.get("new_evaluations_all_datasets") != expected.get("new_evaluations_per_dataset", 0) * 9:
    raise SystemExit("[ABORT] Completion counts do not multiply over nine datasets")
root = pathlib.Path(datasets_doc["dataset_root"])
for order, item in enumerate(datasets, start=1):
    if item.get("order") != order:
        raise SystemExit("[ABORT] Dataset order changed")
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

read -r FULL_C2F REUSED_C2F NEW_C2F FULL_DIRECT REUSED_DIRECT NEW_DIRECT NEW_PER_DATASET NEW_TOTAL < <(
    "$PYTHON" - "$COMPLETION_PLAN" <<'PY'
import json
import pathlib
import sys
x=json.loads(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'))['expected']
print(x['full_c2f_pairs'],x['reused_step_q_c2f_pairs'],x['new_c2f_pairs'],x['full_direct_parents'],x['reused_step_q_direct_parents'],x['new_direct_parents'],x['new_evaluations_per_dataset'],x['new_evaluations_all_datasets'])
PY
)

printf '%s\n' '=============================================================================='
printf '%s STEP-R: REDUCED COMPLETE C2F GRID (MISSING CELLS ONLY)\n' "${ARCHITECTURE^^}"
printf '%s\n' '=============================================================================='
printf '[MODE] %s\n' "$MODE"
printf '[GRID] %s C2F pairs (A/B) + %s direct parents; no gray baseline\n' "$FULL_C2F" "$FULL_DIRECT"
printf '[REUSE] Step-Q read-only: %s C2F + %s direct parents per dataset\n' "$REUSED_C2F" "$REUSED_DIRECT"
printf '[NEW] Step-R: %s C2F + %s direct = %s per dataset / %s total evaluations\n' "$NEW_C2F" "$NEW_DIRECT" "$NEW_PER_DATASET" "$NEW_TOTAL"
printf '[MAPPING] gray + sensor depth; only tracking direct/C2F feature configuration changes\n'
printf '[METRIC] historical keyframe evo_ape ATE mean (--align --correct_scale); all diagnostics retained\n'
printf '[TIMEOUT] %ds per run\n' "$TIMEOUT_SECONDS"
printf '[OUTPUT] %s\n' "$OUTPUT_DIR"
for row in "${DATASET_ROWS[@]}"; do
    IFS=$'\t' read -r key path frames family condition <<< "$row"
    printf '  [%s] frames=%s family=%s condition=%s\n' "$key" "$frames" "$family" "$condition"
done

if [[ "$MODE" == "aggregate" ]]; then
    "$PYTHON" "$AGGREGATOR" --architecture "$ARCHITECTURE" --output-dir "$OUTPUT_DIR" --completion-plan "$COMPLETION_PLAN"
    exit 0
fi

if [[ "$MODE" == "validate" ]]; then
    for row in "${DATASET_ROWS[@]}"; do
        IFS=$'\t' read -r key path _frames _family _condition <<< "$row"
        "$PYTHON" "$EVALUATOR" --architecture "$ARCHITECTURE" --dataset-dir "$path" --completion-plan "$COMPLETION_PLAN" --timeout-seconds "$TIMEOUT_SECONDS"
        printf '[VALID] %s\n' "$key"
    done
    printf '\n[VALIDATION COMPLETE] Step-Q reuse rows were checked read-only; no COMO process, new SQLite row, trajectory, or config edit was created.\n'
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
exec 9>"$RESULTS_ROOT/.c2f_grid_completion_global.lock"
flock -n 9 || die 'Another Step-R C2F completion launcher is active; do not run architectures concurrently.'
stamp="$(date '+%Y%m%d_%H%M%S')"
backup_dir="$OUTPUT_DIR/launch_backups/$stamp"
mkdir -p "$backup_dir/databases"
cp -- "$CONFIG" "$backup_dir/como_before_launch.yml"
cp -- "$COMPLETION_PLAN" "$backup_dir/${ARCHITECTURE}_c2f_grid_completion_plan.json"

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
printf '[BACKUP] Step-R launch snapshot: %s\n' "$backup_dir"

for row in "${DATASET_ROWS[@]}"; do
    IFS=$'\t' read -r key path frames family condition <<< "$row"
    dataset_output="$OUTPUT_DIR/per_dataset/$key"
    printf '\n[DATASET START] %s (%s frames; %s/%s)\n' "$key" "$frames" "$family" "$condition"
    taskset -c "$CPU_AFFINITY" "$PYTHON" "$EVALUATOR" --architecture "$ARCHITECTURE" --execute \
        --dataset-dir "$path" --output-dir "$dataset_output" --completion-plan "$COMPLETION_PLAN" --timeout-seconds "$TIMEOUT_SECONDS"
    "$PYTHON" "$AGGREGATOR" --architecture "$ARCHITECTURE" --output-dir "$OUTPUT_DIR" --completion-plan "$COMPLETION_PLAN"
done

"$PYTHON" "$AGGREGATOR" --architecture "$ARCHITECTURE" --output-dir "$OUTPUT_DIR" --completion-plan "$COMPLETION_PLAN"
printf '\n[DONE] %s Step-R grid completion reached its current resumable state.\n' "$ARCHITECTURE"
printf '[READ FIRST] %s/aggregate_summary.md\n' "$OUTPUT_DIR"
printf '[MERGED EVIDENCE] %s/merged_pairwise_comparison.csv\n' "$OUTPUT_DIR"
