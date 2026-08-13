#!/usr/bin/env bash
# Resumable 7 QueensCAMP degradations × (Top-7 + gray) × 3 runs.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
PYTHON="/home/melody/anaconda3/envs/como/bin/python"
EVALUATOR="$SCRIPT_DIR/../step_e_full_sequence_evaluation/run_full_sequence_evaluation.py"
AGGREGATOR="$SCRIPT_DIR/aggregate_queenscamp_3x_results.py"
EXPERIMENT_PLAN="$SCRIPT_DIR/queenscamp_3x_plan.json"
CANDIDATE_PLAN="$SCRIPT_DIR/top7_plus_gray_candidate_plan.json"
OUTPUT="$PROJECT_ROOT/channel_selection_results/step_h_queenscamp_degradation_evaluation/three_repeats"
CONFIG="$PROJECT_ROOT/como/config/como.yml"
CPU_AFFINITY="${CPU_AFFINITY:-0,1,4-9,12-15}"
NO_TURBO_PATH="/sys/devices/system/cpu/intel_pstate/no_turbo"
export MPLCONFIGDIR="$OUTPUT/.matplotlib"
export XDG_CACHE_HOME="$OUTPUT/.cache"

usage() {
    printf 'Usage: %s [--execute|--aggregate-only]\n' "$0"
    printf '  no argument       validate inputs and dry-run all dataset plans\n'
    printf '  --execute         run/resume all 168 evaluations\n'
    printf '  --aggregate-only  rebuild tables/plots from saved databases\n'
}

mode="dry-run"
if (( $# > 1 )); then usage >&2; exit 2; fi
if (( $# == 1 )); then
    case "$1" in
        --execute) mode="execute" ;;
        --aggregate-only) mode="aggregate" ;;
        --help|-h) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
fi

for required in "$PYTHON" "$EVALUATOR" "$AGGREGATOR" "$EXPERIMENT_PLAN" \
    "$CANDIDATE_PLAN" "$CONFIG"; do
    [[ -e "$required" ]] || { printf '[ABORT] Missing required path: %s\n' "$required" >&2; exit 1; }
done

mapfile -t DATASETS < <(
    "$PYTHON" - "$EXPERIMENT_PLAN" "$CANDIDATE_PLAN" <<'PY'
import json
import pathlib
import sys

experiment = json.loads(pathlib.Path(sys.argv[1]).read_text())
candidate_plan = json.loads(pathlib.Path(sys.argv[2]).read_text())
candidates = candidate_plan["candidates"]
assert experiment["protocol"] == "top7_plus_gray_queenscamp_3x_v1"
assert candidate_plan["protocol"] == "top7_plus_gray_candidate_plan_v1"
assert experiment["planned_runs"] == 168
assert experiment["replicates_per_configuration"] == 3
assert experiment["timeout_seconds_per_run"] == 500
assert len(experiment["datasets"]) == 7 and len(candidates) == 8
expected_keys = [
    "5,6,24,29",
    "1,26,30,40",
    "15,17,52,59",
    "1,5,24,29",
    "5,6,15,35",
    "6,10,34,41",
    "5,29,40,52",
    "gray",
]
assert [item["candidate_key"] for item in candidates] == expected_keys
assert candidates[-1]["channels"] is None and candidates[-1]["label"] == "gray_baseline"
root = pathlib.Path(experiment["dataset_root"])

def index_rows(path):
    rows = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) < 2:
            raise AssertionError(f"Malformed row in {path}: {line!r}")
        rows.append((float(fields[0]), fields[1]))
    if any(b[0] <= a[0] for a, b in zip(rows, rows[1:])):
        raise AssertionError(f"Non-increasing timestamps in {path}")
    return rows

for expected_order, item in enumerate(experiment["datasets"], start=1):
    assert item["order"] == expected_order
    dataset = root / item["directory_name"]
    for name in ("matched_rgb.txt", "matched_depth.txt", "groundtruth.txt"):
        assert (dataset / name).is_file(), dataset / name
    rgb_rows = index_rows(dataset / "matched_rgb.txt")
    depth_rows = index_rows(dataset / "matched_depth.txt")
    expected = item["expected_matched_frames"]
    assert len(rgb_rows) == expected, (item["key"], "rgb", len(rgb_rows), expected)
    assert len(depth_rows) == expected, (item["key"], "depth", len(depth_rows), expected)
    for _, relative in rgb_rows:
        assert (dataset / relative).is_file(), dataset / relative
    for _, relative in depth_rows:
        assert (dataset / relative).is_file(), dataset / relative
    print(f"{item['key']}\t{dataset}\t{len(rgb_rows)}\t{item['degradation']}")
PY
)

printf '%s\n' '=============================================================================='
printf '%s\n' 'QUEENSCAMP DEGRADATION ROBUSTNESS: TOP-7 + GRAY × THREE REPEATS'
printf '%s\n' '=============================================================================='
printf '[MODE] %s\n' "$mode"
printf '[PLAN] 7 datasets × 8 configurations × 3 runs = 168 runs\n'
printf '[METRIC] keyframe evo_ape ATE mean with --align --correct_scale\n'
printf '[DIAGNOSTICS] full-frame SE(3) ATE/RPE, keyframe RPE, coverage, failures\n'
printf '[TIMEOUT] 500 seconds per run\n'
printf '[OUTPUT] %s\n' "$OUTPUT"
for row in "${DATASETS[@]}"; do printf '  %s\n' "$row"; done

mkdir -p "$OUTPUT/per_dataset" "$OUTPUT/launch_backups" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
if [[ "$mode" == "aggregate" ]]; then
    "$PYTHON" "$AGGREGATOR" \
        --output-dir "$OUTPUT" \
        --experiment-plan "$EXPERIMENT_PLAN" \
        --candidate-plan "$CANDIDATE_PLAN"
    exit 0
fi

backup_dir=""
restore_config() {
    if [[ -n "$backup_dir" && -f "$backup_dir/como_before_launch.yml" ]]; then
        cp -- "$backup_dir/como_before_launch.yml" "$CONFIG"
    fi
}

if [[ "$mode" == "execute" ]]; then
    command -v nvidia-smi >/dev/null || { printf '[ABORT] nvidia-smi unavailable\n' >&2; exit 1; }
    taskset -c "$CPU_AFFINITY" true 2>/dev/null || { printf '[ABORT] Invalid CPU affinity: %s\n' "$CPU_AFFINITY" >&2; exit 1; }
    if [[ -r "$NO_TURBO_PATH" && "$(<"$NO_TURBO_PATH")" != "1" ]]; then
        printf '[ABORT] Intel Turbo is enabled; expected no_turbo=1 for the established stable setup\n' >&2
        exit 1
    fi
    stamp="$(date '+%Y%m%d_%H%M%S')"
    backup_dir="$OUTPUT/launch_backups/$stamp"
    mkdir -p "$backup_dir/databases"
    cp -- "$CONFIG" "$backup_dir/como_before_launch.yml"
    cp -- "$EXPERIMENT_PLAN" "$backup_dir/queenscamp_3x_plan.json"
    cp -- "$CANDIDATE_PLAN" "$backup_dir/top7_plus_gray_candidate_plan.json"
    for row in "${DATASETS[@]}"; do
        IFS=$'\t' read -r key _dataset _frames _degradation <<< "$row"
        database="$OUTPUT/per_dataset/$key/evaluations.sqlite3"
        if [[ -f "$database" ]]; then
            "$PYTHON" - "$database" "$backup_dir/databases/$key.sqlite3" <<'PY'
import sqlite3
import sys
source = sqlite3.connect(sys.argv[1])
target = sqlite3.connect(sys.argv[2])
source.backup(target)
target.close()
source.close()
PY
        fi
    done
    trap restore_config EXIT
    nvidia-smi -L
fi

for row in "${DATASETS[@]}"; do
    IFS=$'\t' read -r key dataset frames degradation <<< "$row"
    printf '\n[DATASET START] %s degradation=%s frames=%s\n' "$key" "$degradation" "$frames"
    args=(
        --candidate-plan "$CANDIDATE_PLAN"
        --dataset-dir "$dataset"
        --output-dir "$OUTPUT/per_dataset/$key"
        --replicates 3
        --timeout-seconds 500
    )
    if [[ "$mode" == "execute" ]]; then args=(--execute "${args[@]}"); fi
    taskset -c "$CPU_AFFINITY" "$PYTHON" "$EVALUATOR" "${args[@]}"
    if [[ "$mode" == "execute" ]]; then
        "$PYTHON" "$AGGREGATOR" \
            --output-dir "$OUTPUT" \
            --experiment-plan "$EXPERIMENT_PLAN" \
            --candidate-plan "$CANDIDATE_PLAN"
    fi
done

"$PYTHON" "$AGGREGATOR" \
    --output-dir "$OUTPUT" \
    --experiment-plan "$EXPERIMENT_PLAN" \
    --candidate-plan "$CANDIDATE_PLAN"

if [[ "$mode" == "dry-run" ]]; then
    printf '\n[DRY RUN COMPLETE] Input files and 168-run plan validated; COMO was not launched.\n'
    printf '[NEXT] Add --execute to run or resume the experiment.\n'
else
    printf '\n[DONE] Experiment finished or resumed through all available runs.\n'
    printf '[SUMMARY] %s/aggregate_summary.md\n' "$OUTPUT"
fi
