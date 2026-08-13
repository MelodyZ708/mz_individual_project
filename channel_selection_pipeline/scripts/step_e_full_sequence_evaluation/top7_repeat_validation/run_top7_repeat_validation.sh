#!/usr/bin/env bash
# Independent full-sequence repeat of the six second-stage winners plus baseline,
# followed by feature-map visualisation and r=0.70 cluster attribution.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../../..")"
PYTHON="/home/melody/anaconda3/envs/como/bin/python"
EVO_APE="/home/melody/anaconda3/envs/como/bin/evo_ape"
EVALUATOR="$SCRIPT_DIR/../run_full_sequence_evaluation.py"
ANALYSER="$SCRIPT_DIR/analyze_top7_repeat.py"
PLAN="$SCRIPT_DIR/top7_candidate_plan.json"
OUTPUT_DIR="$PROJECT_ROOT/channel_selection_results/step_e_full_sequence_evaluation/top7_repeat_feature_cluster_analysis"
REFERENCE_DB="$PROJECT_ROOT/channel_selection_results/step_e_full_sequence_evaluation/second_round_baseline_plus2_rpe_safe/evaluations.sqlite3"
CONFIG="$PROJECT_ROOT/como/config/como.yml"
CPU_AFFINITY="${CPU_AFFINITY:-0,1,4-9,12-15}"
NO_TURBO_PATH="/sys/devices/system/cpu/intel_pstate/no_turbo"

die() {
    printf '[ABORT] %s\n' "$*" >&2
    exit 1
}

execute_requested=0
for argument in "$@"; do
    if [[ "$argument" == "--execute" ]]; then
        execute_requested=1
    fi
done

for required in "$PYTHON" "$EVO_APE" "$EVALUATOR" "$ANALYSER" \
    "$PLAN" "$REFERENCE_DB" "$CONFIG"; do
    [[ -e "$required" ]] || die "Required path is missing: $required"
done

"$PYTHON" - "$PLAN" "$REFERENCE_DB" <<'PY'
import json
import sqlite3
import sys

plan_path, database_path = sys.argv[1:]
plan = json.load(open(plan_path, encoding="utf-8"))
candidates = plan.get("candidates", [])
expected = [
    "5,6,24,29",
    "1,26,30,40",
    "15,17,52,59",
    "1,5,24,29",
    "5,6,15,35",
    "6,10,34,41",
    "5,29,40,52",
]
actual = [item.get("candidate_key") for item in candidates]
if plan.get("protocol") != "full_sequence_top7_independent_repeat_v1":
    raise SystemExit("[ABORT] Unexpected Top-7 protocol")
if actual != expected or plan.get("selection", {}).get("selected_count") != 7:
    raise SystemExit("[ABORT] Frozen Top-7 candidate order/content changed")
connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
    raise SystemExit("[ABORT] Second-stage database integrity check failed")
rows = dict(
    connection.execute(
        "SELECT candidate_key, status FROM evaluations WHERE replicate=0"
    )
)
connection.close()
invalid = [key for key in expected if rows.get(key) != "PASS"]
if invalid:
    raise SystemExit("[ABORT] Top-7 reference PASS rows missing: " + str(invalid))
print("[CHECK] Frozen population: second-stage ranks 1--6 plus rank-7 baseline")
print("[CHECK] Reference database integrity: ok; all seven original rows are PASS")
PY

printf '%s\n' '=============================================================================='
printf '%s\n' 'TOP-7 INDEPENDENT FULL-SEQUENCE REPEAT + FEATURE/CLUSTER ANALYSIS'
printf '%s\n' '=============================================================================='
printf '[PLAN] Seven new COMO runs on complete fr1/desk_lightswitch\n'
printf '[PLAN] Primary ATE: historical keyframe evo_ape mean (--align --correct_scale)\n'
printf '[PLAN] Per-run timeout: 300 seconds\n'
printf '[PLAN] Feature frames: original indices 246 before, 250 peak, 254 after\n'
printf '[PLAN] Feature view: Conv1 post-ReLU clean, lightswitch and absolute difference\n'
printf '[PLAN] Cluster attribution: final global r=0.70 Conv1 clusters\n'
printf '[PLAN] Output: %s\n' "$OUTPUT_DIR"

if (( execute_requested == 1 )); then
    command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable"
    command -v taskset >/dev/null 2>&1 || die "taskset is unavailable"
    taskset -c "$CPU_AFFINITY" true 2>/dev/null || \
        die "Invalid or unavailable CPU affinity: $CPU_AFFINITY"
    if [[ -r "$NO_TURBO_PATH" && "$(<"$NO_TURBO_PATH")" != "1" ]]; then
        die "Intel Turbo is enabled; disable it before this repeat for comparability"
    fi
    if command -v journalctl >/dev/null 2>&1; then
        hardware_events="$(
            journalctl -b 0 -k --no-pager 2>/dev/null | grep -Ei \
                'mce: \[Hardware Error\]|machine check events logged|NVRM: Xid|GPU has fallen off|oom-kill' \
                || true
        )"
        [[ -z "$hardware_events" ]] || {
            printf '%s\n' "$hardware_events" >&2
            die "Current boot already contains a matching hardware/driver event"
        }
    fi
    printf '[CHECK] GPU: '
    nvidia-smi -L || die "NVIDIA driver/GPU is unavailable"
fi

mkdir -p "$OUTPUT_DIR/launch_backups"
if (( execute_requested == 1 )); then
    stamp="$(date '+%Y%m%d_%H%M%S')"
    backup_dir="$OUTPUT_DIR/launch_backups/$stamp"
    mkdir -p "$backup_dir"
    cp -- "$CONFIG" "$backup_dir/como_before_launch.yml"
    cp -- "$PLAN" "$backup_dir/top7_candidate_plan.json"
    database="$OUTPUT_DIR/evaluations.sqlite3"
    if [[ -f "$database" ]]; then
        "$PYTHON" - "$database" "$backup_dir/evaluations_before_launch.sqlite3" <<'PY'
import sqlite3
import sys

source_path, backup_path = sys.argv[1:]
source = sqlite3.connect(f"file:{source_path}?mode=ro", uri=True)
if source.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
    raise SystemExit("[ABORT] Existing Top-7 repeat DB integrity check failed")
target = sqlite3.connect(backup_path)
source.backup(target)
target.close()
source.close()
print(f"[BACKUP] Existing repeat DB: {backup_path}")
PY
    fi
fi

printf '[LAUNCH] Evaluator arguments:'
printf ' %q' "$@"
printf '\n\n'

taskset -c "$CPU_AFFINITY" "$PYTHON" "$EVALUATOR" \
    --candidate-plan "$PLAN" \
    --output-dir "$OUTPUT_DIR" \
    --evo-ape "$EVO_APE" \
    --timeout-seconds 300 \
    "$@"

if (( execute_requested == 0 )); then
    printf '\n[DRY RUN] No COMO run or analysis was executed. Add --execute after reviewing the plan.\n'
    exit 0
fi

printf '\n[ANALYSIS] Building repeat comparison, feature maps and cluster tables...\n'
"$PYTHON" "$ANALYSER" \
    --plan "$PLAN" \
    --output-dir "$OUTPUT_DIR" \
    --reference-db "$REFERENCE_DB"

printf '\n[DONE] Top-7 independent repeat and analysis are complete.\n'
printf '[RESULTS] %s\n' "$OUTPUT_DIR"
printf '[READ FIRST] %s/summary.md\n' "$OUTPUT_DIR"
printf '[TABLE] %s/repeat_comparison.csv\n' "$OUTPUT_DIR"
printf '[FEATURE MAPS] %s/feature_maps/lightswitch_overviews/\n' "$OUTPUT_DIR"
