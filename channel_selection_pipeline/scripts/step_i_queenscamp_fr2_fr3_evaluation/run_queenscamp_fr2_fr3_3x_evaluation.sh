#!/usr/bin/env bash
# Resumable fr2/fr3 QueensCAMP evaluation: 14 datasets × (Top-7 + gray) × 3.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
PYTHON="/home/melody/anaconda3/envs/como/bin/python"
EVALUATOR="$SCRIPT_DIR/../step_e_full_sequence_evaluation/run_full_sequence_evaluation.py"
AGGREGATOR="$SCRIPT_DIR/aggregate_queenscamp_fr2_fr3_3x_results.py"
EXPERIMENT_PLAN="$SCRIPT_DIR/queenscamp_fr2_fr3_3x_plan.json"
CANDIDATE_PLAN="$SCRIPT_DIR/top7_plus_gray_candidate_plan.json"
OUTPUT="$PROJECT_ROOT/channel_selection_results/step_i_queenscamp_fr2_fr3_evaluation/three_repeats"
CONFIG="$PROJECT_ROOT/como/config/como.yml"
CPU_AFFINITY="${CPU_AFFINITY:-0,1,4-9,12-15}"
NO_TURBO_PATH="/sys/devices/system/cpu/intel_pstate/no_turbo"
export MPLCONFIGDIR="$OUTPUT/.matplotlib"
export XDG_CACHE_HOME="$OUTPUT/.cache"

usage() {
    printf 'Usage: %s [--execute|--aggregate-only]\n' "$0"
    printf '  no argument       validate all inputs and dry-run the 336-run plan\n'
    printf '  --execute         run/resume every missing replicate\n'
    printf '  --aggregate-only  rebuild tables and plots from saved databases\n'
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

for required in "$PYTHON" "$EVALUATOR" "$AGGREGATOR" "$EXPERIMENT_PLAN" "$CANDIDATE_PLAN" "$CONFIG"; do
    [[ -e "$required" ]] || { printf '[ABORT] Missing required path: %s\n' "$required" >&2; exit 1; }
done

mapfile -t DATASETS < <(
    "$PYTHON" - "$EXPERIMENT_PLAN" "$CANDIDATE_PLAN" <<'PY'
import json
import pathlib
import sys

plan = json.loads(pathlib.Path(sys.argv[1]).read_text())
candidates = json.loads(pathlib.Path(sys.argv[2]).read_text())["candidates"]
assert plan["protocol"] == "top7_plus_gray_queenscamp_fr2_fr3_3x_v1"
assert plan["family_count"] == 2 and plan["dataset_count"] == 14
assert plan["configuration_count"] == 8 and plan["replicates_per_configuration"] == 3
assert plan["timeout_seconds_per_run"] == 500 and plan["planned_runs"] == 336
assert [x["candidate_key"] for x in candidates] == [
    "5,6,24,29", "1,26,30,40", "15,17,52,59", "1,5,24,29",
    "5,6,15,35", "6,10,34,41", "5,29,40,52", "gray"]
assert candidates[-1]["channels"] is None
root = pathlib.Path(plan["dataset_root"])

def index_rows(path):
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            fields = line.split()
            assert len(fields) >= 2, (path, line)
            rows.append((float(fields[0]), fields[1]))
    assert all(b[0] > a[0] for a, b in zip(rows, rows[1:])), path
    return rows

families = {"fr2_desk": 2893, "fr3_long_office_household": 2488}
assert len(plan["datasets"]) == 14
for order, item in enumerate(plan["datasets"], 1):
    assert item["order"] == order and item["family"] in families
    assert item["expected_matched_frames"] == families[item["family"]]
    dataset = root / item["directory_name"]
    for name in ("matched_rgb.txt", "matched_depth.txt", "groundtruth.txt"):
        assert (dataset / name).is_file(), dataset / name
    rgb, depth = index_rows(dataset / "matched_rgb.txt"), index_rows(dataset / "matched_depth.txt")
    assert len(rgb) == len(depth) == item["expected_matched_frames"], item["key"]
    for _, relative in rgb + depth:
        assert (dataset / relative).is_file(), dataset / relative
    print(f"{item['key']}\t{dataset}\t{len(rgb)}\t{item['family']}\t{item['degradation']}")
PY
)

printf '%s\n' '=============================================================================='
printf '%s\n' 'QUEENSCAMP FR2/FR3 ROBUSTNESS: TOP-7 + GRAY × THREE REPEATS'
printf '%s\n' '=============================================================================='
printf '[MODE] %s\n' "$mode"
printf '[PLAN] 14 datasets (7 fr2 + 7 fr3) × 8 configurations × 3 runs = 336 runs\n'
printf '[METRIC] historical keyframe evo_ape ATE mean with --align --correct_scale\n'
printf '[DIAGNOSTICS] full-frame SE(3) ATE/RPE, keyframe RPE, coverage, failures\n'
printf '[TIMEOUT] 500 seconds per run\n[OUTPUT] %s\n' "$OUTPUT"
printf '%s\n' "${DATASETS[@]}"

mkdir -p "$OUTPUT/per_dataset" "$OUTPUT/launch_backups" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
if [[ "$mode" == "aggregate" ]]; then
    exec "$PYTHON" "$AGGREGATOR" --output-dir "$OUTPUT" --experiment-plan "$EXPERIMENT_PLAN" --candidate-plan "$CANDIDATE_PLAN"
fi

backup_dir=""
restore_config() {
    [[ -n "$backup_dir" && -f "$backup_dir/como_before_launch.yml" ]] && cp -- "$backup_dir/como_before_launch.yml" "$CONFIG"
}
if [[ "$mode" == "execute" ]]; then
    command -v nvidia-smi >/dev/null || { printf '[ABORT] nvidia-smi unavailable\n' >&2; exit 1; }
    taskset -c "$CPU_AFFINITY" true 2>/dev/null || { printf '[ABORT] Invalid CPU affinity: %s\n' "$CPU_AFFINITY" >&2; exit 1; }
    if [[ -r "$NO_TURBO_PATH" && "$(<"$NO_TURBO_PATH")" != "1" ]]; then
        printf '[ABORT] Intel Turbo is enabled; expected no_turbo=1 for the established stable setup\n' >&2; exit 1
    fi
    backup_dir="$OUTPUT/launch_backups/$(date '+%Y%m%d_%H%M%S')"
    mkdir -p "$backup_dir/databases"
    cp -- "$CONFIG" "$backup_dir/como_before_launch.yml"
    cp -- "$EXPERIMENT_PLAN" "$backup_dir/experiment_plan.json"
    cp -- "$CANDIDATE_PLAN" "$backup_dir/candidate_plan.json"
    for row in "${DATASETS[@]}"; do
        IFS=$'\t' read -r key _dataset _frames _family _degradation <<< "$row"
        database="$OUTPUT/per_dataset/$key/evaluations.sqlite3"
        if [[ -f "$database" ]]; then
            "$PYTHON" - "$database" "$backup_dir/databases/$key.sqlite3" <<'PY'
import sqlite3, sys
source, target = sqlite3.connect(sys.argv[1]), sqlite3.connect(sys.argv[2])
source.backup(target); target.close(); source.close()
PY
        fi
    done
    trap restore_config EXIT
    nvidia-smi -L
fi

for row in "${DATASETS[@]}"; do
    IFS=$'\t' read -r key dataset frames family degradation <<< "$row"
    printf '\n[DATASET START] %s (%s; %s; %s frames)\n' "$key" "$family" "$degradation" "$frames"
    args=(--candidate-plan "$CANDIDATE_PLAN" --dataset-dir "$dataset" --output-dir "$OUTPUT/per_dataset/$key" --replicates 3 --timeout-seconds 500)
    [[ "$mode" == "execute" ]] && args=(--execute "${args[@]}")
    taskset -c "$CPU_AFFINITY" "$PYTHON" "$EVALUATOR" "${args[@]}"
    [[ "$mode" == "execute" ]] && "$PYTHON" "$AGGREGATOR" --output-dir "$OUTPUT" --experiment-plan "$EXPERIMENT_PLAN" --candidate-plan "$CANDIDATE_PLAN"
done

"$PYTHON" "$AGGREGATOR" --output-dir "$OUTPUT" --experiment-plan "$EXPERIMENT_PLAN" --candidate-plan "$CANDIDATE_PLAN"
if [[ "$mode" == "dry-run" ]]; then
    printf '\n[DRY RUN COMPLETE] All 14 datasets and the 336-run plan were validated; COMO was not launched.\n'
    printf '[NEXT] Add --execute to run or resume.\n'
else
    printf '\n[DONE] Results: %s/aggregate_summary.md\n' "$OUTPUT"
fi
