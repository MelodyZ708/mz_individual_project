#!/usr/bin/env bash
# Resumable 3 lightswitch datasets × Top-7 × 5 independent runs.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
PYTHON="/home/melody/anaconda3/envs/como/bin/python"
EVALUATOR="$SCRIPT_DIR/../step_e_full_sequence_evaluation/run_full_sequence_evaluation.py"
AGGREGATOR="$SCRIPT_DIR/aggregate_lightswitch_5x_results.py"
EXPERIMENT_PLAN="$SCRIPT_DIR/lightswitch_5x_plan.json"
CANDIDATE_PLAN="$SCRIPT_DIR/top7_candidate_plan.json"
OUTPUT="$PROJECT_ROOT/channel_selection_results/step_f_multi_dataset_evaluation/lightswitch_5x_evaluation"
CONFIG="$PROJECT_ROOT/como/config/como.yml"
CPU_AFFINITY="${CPU_AFFINITY:-0,1,4-9,12-15}"
NO_TURBO_PATH="/sys/devices/system/cpu/intel_pstate/no_turbo"
export MPLCONFIGDIR="$OUTPUT/.matplotlib"
export XDG_CACHE_HOME="$OUTPUT/.cache"

usage() {
    printf 'Usage: %s [--execute|--aggregate-only]\n' "$0"
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

mapfile -t DATASETS < <(
    "$PYTHON" - "$EXPERIMENT_PLAN" "$CANDIDATE_PLAN" <<'PY'
import json,pathlib,sys
plan=json.loads(pathlib.Path(sys.argv[1]).read_text())
candidates=json.loads(pathlib.Path(sys.argv[2]).read_text())['candidates']
assert plan['protocol']=='top7_lightswitch_5x_v1'
assert plan['planned_runs']==105 and plan['replicates_per_candidate']==5
assert len(plan['datasets'])==3 and len(candidates)==7
root=pathlib.Path(plan['dataset_root'])
for item in plan['datasets']:
    path=root/item['directory_name']
    for name in ('matched_rgb.txt','matched_depth.txt','groundtruth.txt'):
        assert (path/name).is_file(), path/name
    count=sum(1 for line in (path/'matched_rgb.txt').open() if line.strip() and not line.lstrip().startswith('#'))
    assert count==item['expected_matched_frames'], (item['key'],count)
    print(f"{item['key']}\t{path}\t{count}")
PY
)

printf '%s\n' '=============================================================================='
printf '%s\n' 'THREE LIGHTSWITCH DATASETS × TOP-7 × FIVE REPEATS'
printf '%s\n' '=============================================================================='
printf '[MODE] %s\n' "$mode"
printf '[PLAN] 3 datasets × 7 configurations × 5 runs = 105 runs\n'
printf '[PLAN] timeout=500s; means/std/medians computed over PASS observations only\n'
printf '[OUTPUT] %s\n' "$OUTPUT"
for row in "${DATASETS[@]}"; do printf '  %s\n' "$row"; done

mkdir -p "$OUTPUT/per_dataset" "$OUTPUT/launch_backups" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
if [[ "$mode" == "aggregate" ]]; then
    "$PYTHON" "$AGGREGATOR" --output-dir "$OUTPUT" --experiment-plan "$EXPERIMENT_PLAN" --candidate-plan "$CANDIDATE_PLAN"
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
    taskset -c "$CPU_AFFINITY" true 2>/dev/null || { printf '[ABORT] invalid CPU affinity\n' >&2; exit 1; }
    if [[ -r "$NO_TURBO_PATH" && "$(<"$NO_TURBO_PATH")" != "1" ]]; then
        printf '[ABORT] Intel Turbo is enabled\n' >&2; exit 1
    fi
    stamp="$(date '+%Y%m%d_%H%M%S')"
    backup_dir="$OUTPUT/launch_backups/$stamp"
    mkdir -p "$backup_dir"
    cp -- "$CONFIG" "$backup_dir/como_before_launch.yml"
    cp -- "$EXPERIMENT_PLAN" "$backup_dir/lightswitch_5x_plan.json"
    cp -- "$CANDIDATE_PLAN" "$backup_dir/top7_candidate_plan.json"
    trap restore_config EXIT
    nvidia-smi -L
fi

for row in "${DATASETS[@]}"; do
    IFS=$'\t' read -r key dataset frames <<< "$row"
    printf '\n[DATASET START] %s frames=%s\n' "$key" "$frames"
    args=(
        --candidate-plan "$CANDIDATE_PLAN"
        --dataset-dir "$dataset"
        --output-dir "$OUTPUT/per_dataset/$key"
        --replicates 5
        --timeout-seconds 500
    )
    if [[ "$mode" == "execute" ]]; then args=(--execute "${args[@]}"); fi
    taskset -c "$CPU_AFFINITY" "$PYTHON" "$EVALUATOR" "${args[@]}"
    "$PYTHON" "$AGGREGATOR" --output-dir "$OUTPUT" --experiment-plan "$EXPERIMENT_PLAN" --candidate-plan "$CANDIDATE_PLAN"
done

if [[ "$mode" == "dry-run" ]]; then
    printf '\n[DRY RUN COMPLETE] COMO was not launched; add --execute to run.\n'
else
    printf '\n[DONE] Read %s/aggregate_summary.md\n' "$OUTPUT"
fi
