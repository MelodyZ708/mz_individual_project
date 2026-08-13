#!/usr/bin/env bash
# Resumable Top-7 evaluation on eight clean/degraded TUM sequences.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
PYTHON="/home/melody/anaconda3/envs/como/bin/python"
EVO_APE="/home/melody/anaconda3/envs/como/bin/evo_ape"
EVO_RPE="/home/melody/anaconda3/envs/como/bin/evo_rpe"
EVALUATOR="$SCRIPT_DIR/../step_e_full_sequence_evaluation/run_full_sequence_evaluation.py"
AGGREGATOR="$SCRIPT_DIR/aggregate_multi_dataset_results.py"
DATASET_PLAN="$SCRIPT_DIR/dataset_plan.json"
CANDIDATE_PLAN="$SCRIPT_DIR/top7_candidate_plan.json"
OUTPUT_DIR="$PROJECT_ROOT/channel_selection_results/step_f_multi_dataset_evaluation"
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
  run_top7_multi_dataset_evaluation.sh             # validate and dry-run only
  run_top7_multi_dataset_evaluation.sh --execute   # run/resume all 56 evaluations
  run_top7_multi_dataset_evaluation.sh --aggregate-only

The same --execute command is resumable. Saved dataset/candidate rows are skipped.
EOF
}

mode="dry-run"
if (( $# > 1 )); then
    usage >&2
    exit 2
fi
if (( $# == 1 )); then
    case "$1" in
        --execute) mode="execute" ;;
        --aggregate-only) mode="aggregate" ;;
        --help|-h) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
fi

for required in "$PYTHON" "$EVO_APE" "$EVO_RPE" "$EVALUATOR" "$AGGREGATOR" \
    "$DATASET_PLAN" "$CANDIDATE_PLAN" "$CONFIG"; do
    [[ -e "$required" ]] || die "Required path is missing: $required"
done

mapfile -t DATASET_ROWS < <(
    "$PYTHON" - "$DATASET_PLAN" "$CANDIDATE_PLAN" <<'PY'
import json
import pathlib
import sys

dataset_plan_path, candidate_plan_path = map(pathlib.Path, sys.argv[1:])
dataset_plan = json.loads(dataset_plan_path.read_text(encoding="utf-8"))
candidate_plan = json.loads(candidate_plan_path.read_text(encoding="utf-8"))
datasets = dataset_plan.get("datasets", [])
candidates = candidate_plan.get("candidates", [])
if dataset_plan.get("protocol") != "top7_multi_dataset_generalisation_v1":
    raise SystemExit("[ABORT] Unexpected dataset protocol")
if candidate_plan.get("protocol") != "top7_multi_dataset_candidate_plan_v1":
    raise SystemExit("[ABORT] Unexpected candidate protocol")
if dataset_plan.get("timeout_seconds_per_run") != 500:
    raise SystemExit("[ABORT] Timeout in dataset plan is not 500 seconds")
if dataset_plan.get("dataset_count") != 8 or len(datasets) != 8:
    raise SystemExit("[ABORT] Expected exactly eight datasets")
if candidate_plan.get("selection", {}).get("selected_count") != 7 or len(candidates) != 7:
    raise SystemExit("[ABORT] Expected exactly seven candidates")
expected_candidates = [
    "5,6,24,29", "1,26,30,40", "15,17,52,59", "1,5,24,29",
    "5,6,15,35", "6,10,34,41", "5,29,40,52",
]
if [item.get("candidate_key") for item in candidates] != expected_candidates:
    raise SystemExit("[ABORT] Frozen Top-7 order/content changed")
root = pathlib.Path(dataset_plan["dataset_root"])
seen = set()
for expected_order, item in enumerate(datasets, 1):
    if item.get("order") != expected_order or item["key"] in seen:
        raise SystemExit("[ABORT] Dataset order/key validation failed")
    seen.add(item["key"])
    path = root / item["directory_name"]
    for filename in ("matched_rgb.txt", "matched_depth.txt", "groundtruth.txt"):
        if not (path / filename).is_file():
            raise SystemExit(f"[ABORT] Missing {path / filename}")
    with (path / "matched_rgb.txt").open(encoding="utf-8") as handle:
        count = sum(1 for line in handle if line.strip() and not line.lstrip().startswith("#"))
    if count != item["expected_matched_frames"]:
        raise SystemExit(
            f"[ABORT] {item['key']} matched frames changed: {count} "
            f"!= {item['expected_matched_frames']}"
        )
    print("\t".join((item["key"], str(path), str(count), item["family"], item["condition"])))
PY
)

(( ${#DATASET_ROWS[@]} == 8 )) || die "Dataset-plan validation returned ${#DATASET_ROWS[@]} rows"

printf '%s\n' '=============================================================================='
printf '%s\n' 'STEP F: TOP-7 MULTI-DATASET GENERALISATION EVALUATION'
printf '%s\n' '=============================================================================='
printf '[MODE] %s\n' "$mode"
printf '[PLAN] 8 datasets × 7 channel configurations = 56 runs\n'
printf '[PLAN] Timeout per run: %d seconds\n' "$TIMEOUT_SECONDS"
printf '[PLAN] Primary metric: keyframe evo_ape ATE mean (--align --correct_scale)\n'
printf '[PLAN] Diagnostics: historical keyframe evo_rpe plus all-frame SE(3) ATE/RPE, coverage and numerical fields\n'
printf '[PLAN] Output root: %s\n' "$OUTPUT_DIR"
printf '[NOTE] fr1/desk_lightswitch is the source-selection sequence and is not rerun or counted in Step F\n'
for row in "${DATASET_ROWS[@]}"; do
    IFS=$'\t' read -r key path frames family condition <<< "$row"
    printf '  [%s] frames=%s family=%s condition=%s path=%s\n' \
        "$key" "$frames" "$family" "$condition" "$path"
done

mkdir -p "$OUTPUT_DIR/per_dataset" "$OUTPUT_DIR/launch_backups"

if [[ "$mode" == "aggregate" ]]; then
    "$PYTHON" "$AGGREGATOR" \
        --output-dir "$OUTPUT_DIR" \
        --dataset-plan "$DATASET_PLAN" \
        --candidate-plan "$CANDIDATE_PLAN"
    exit 0
fi

kernel_hardware_errors() {
    journalctl -b 0 -k --no-pager 2>/dev/null | grep -Ei \
        'mce: \[Hardware Error\]|machine check events logged|EDAC.*(error|uncorrected|corrected)|AER:.*error|NVRM: Xid|GPU has fallen off|out of memory|oom-kill' \
        || true
}

preexisting_hardware_errors=""
backup_dir=""
restore_config() {
    if [[ -n "$backup_dir" && -f "$backup_dir/como_before_launch.yml" ]]; then
        cp -- "$backup_dir/como_before_launch.yml" "$CONFIG"
    fi
}
if [[ "$mode" == "execute" ]]; then
    command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable"
    command -v journalctl >/dev/null 2>&1 || die "journalctl is unavailable"
    command -v taskset >/dev/null 2>&1 || die "taskset is unavailable"
    taskset -c "$CPU_AFFINITY" true 2>/dev/null || \
        die "Invalid or unavailable CPU affinity: $CPU_AFFINITY"
    if [[ -r "$NO_TURBO_PATH" && "$(<"$NO_TURBO_PATH")" != "1" ]]; then
        die "Intel Turbo is enabled; disable it before this long evaluation"
    fi
    preexisting_hardware_errors="$(kernel_hardware_errors)"
    if [[ -n "$preexisting_hardware_errors" ]]; then
        printf '%s\n' "$preexisting_hardware_errors" >&2
        die "Current boot already contains a matching hardware/driver event"
    fi
    printf '[CHECK] GPU: '
    nvidia-smi -L || die "NVIDIA driver/GPU is unavailable"

    stamp="$(date '+%Y%m%d_%H%M%S')"
    backup_dir="$OUTPUT_DIR/launch_backups/$stamp"
    mkdir -p "$backup_dir/databases"
    cp -- "$CONFIG" "$backup_dir/como_before_launch.yml"
    cp -- "$DATASET_PLAN" "$backup_dir/dataset_plan.json"
    cp -- "$CANDIDATE_PLAN" "$backup_dir/top7_candidate_plan.json"
    trap restore_config EXIT
    for row in "${DATASET_ROWS[@]}"; do
        IFS=$'\t' read -r key _rest <<< "$row"
        database="$OUTPUT_DIR/per_dataset/$key/evaluations.sqlite3"
        if [[ -f "$database" ]]; then
            "$PYTHON" - "$database" "$backup_dir/databases/${key}.sqlite3" <<'PY'
import sqlite3
import sys

source_path, backup_path = sys.argv[1:]
source = sqlite3.connect(f"file:{source_path}?mode=ro", uri=True)
if source.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
    raise SystemExit(f"[ABORT] Database integrity check failed: {source_path}")
target = sqlite3.connect(backup_path)
source.backup(target)
target.close()
source.close()
PY
        fi
    done
    printf '[BACKUP] %s\n' "$backup_dir"
fi

for row in "${DATASET_ROWS[@]}"; do
    IFS=$'\t' read -r key path frames family condition <<< "$row"
    dataset_output="$OUTPUT_DIR/per_dataset/$key"
    mkdir -p "$dataset_output"
    printf '\n[DATASET START] %s (%s frames; %s/%s)\n' \
        "$key" "$frames" "$family" "$condition"
    evaluator_args=(
        --candidate-plan "$CANDIDATE_PLAN"
        --dataset-dir "$path"
        --output-dir "$dataset_output"
        --evo-ape "$EVO_APE"
        --evo-rpe "$EVO_RPE"
        --timeout-seconds "$TIMEOUT_SECONDS"
    )
    if [[ "$mode" == "execute" ]]; then
        evaluator_args+=(--execute)
    fi
    taskset -c "$CPU_AFFINITY" "$PYTHON" "$EVALUATOR" "${evaluator_args[@]}"

    if [[ "$mode" == "execute" ]]; then
        current_hardware_errors="$(kernel_hardware_errors)"
        if [[ "$current_hardware_errors" != "$preexisting_hardware_errors" ]]; then
            printf '\n[HARDWARE WARNING] Event set changed after %s:\n%s\n' \
                "$key" "$current_hardware_errors" >&2
            die "Stop before the next dataset and diagnose the new event"
        fi
        "$PYTHON" "$AGGREGATOR" \
            --output-dir "$OUTPUT_DIR" \
            --dataset-plan "$DATASET_PLAN" \
            --candidate-plan "$CANDIDATE_PLAN"
    fi
done

"$PYTHON" "$AGGREGATOR" \
    --output-dir "$OUTPUT_DIR" \
    --dataset-plan "$DATASET_PLAN" \
    --candidate-plan "$CANDIDATE_PLAN"

if [[ "$mode" == "dry-run" ]]; then
    printf '\n[DRY RUN COMPLETE] Plans, paths and evaluator inputs were validated; COMO was not launched.\n'
    printf '[RUN] Add --execute to run/resume all 56 cells.\n'
else
    printf '\n[DONE] Step F evaluation and aggregation completed.\n'
    printf '[READ FIRST] %s/aggregate_summary.md\n' "$OUTPUT_DIR"
    printf '[SCORECARD] %s/dataset_scorecard.csv\n' "$OUTPUT_DIR"
fi
