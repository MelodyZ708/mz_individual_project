#!/usr/bin/env bash
# Independently repeat the three fr2/lightswitch failures without changing Step F.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
PYTHON="/home/melody/anaconda3/envs/como/bin/python"
EVALUATOR="$SCRIPT_DIR/../step_e_full_sequence_evaluation/run_full_sequence_evaluation.py"
COMPARATOR="$SCRIPT_DIR/compare_fr2_lightswitch_rerun.py"
CANDIDATE_PLAN="$SCRIPT_DIR/top7_candidate_plan.json"
DATASET="/home/melody/data/tum/rgbd_dataset_freiburg2_desk_lightswitch"
OUTPUT="$PROJECT_ROOT/channel_selection_results/step_f_multi_dataset_evaluation/repeat_checks/fr2_desk_lightswitch_failed_rerun"
CPU_AFFINITY="${CPU_AFFINITY:-0,1,4-9,12-15}"

mode="dry-run"
if (( $# > 1 )); then
    printf 'Usage: %s [--execute]\n' "$0" >&2
    exit 2
fi
if (( $# == 1 )); then
    [[ "$1" == "--execute" ]] || {
        printf 'Usage: %s [--execute]\n' "$0" >&2
        exit 2
    }
    mode="execute"
fi

printf '%s\n' 'FR2 LIGHTSWITCH FAILURE REPEAT CHECK'
printf '[MODE] %s\n' "$mode"
printf '[PLAN] Rerun only [1,26,30,40], [1,5,24,29], baseline [5,29,40,52]\n'
printf '[PLAN] One independent observation each; timeout=500s; original rows unchanged\n'
printf '[OUTPUT] %s\n' "$OUTPUT"

args=(
    --candidate-plan "$CANDIDATE_PLAN"
    --dataset-dir "$DATASET"
    --output-dir "$OUTPUT"
    --timeout-seconds 500
    --only
    full_rank_02_ch_1_26_30_40
    full_rank_04_ch_1_5_24_29
    full_rank_07_baseline_ch_5_29_40_52
)
if [[ "$mode" == "execute" ]]; then
    args=(--execute "${args[@]}")
fi

taskset -c "$CPU_AFFINITY" "$PYTHON" "$EVALUATOR" "${args[@]}"
"$PYTHON" "$COMPARATOR"

if [[ "$mode" == "dry-run" ]]; then
    printf '[DRY RUN COMPLETE] COMO was not launched; add --execute to run.\n'
else
    printf '[DONE] Read %s/comparison_with_original.md\n' "$OUTPUT"
fi
