#!/usr/bin/env bash
# Safe launcher for the resumable UNet enc1 full-sequence greedy search.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
PYTHON="/home/melody/anaconda3/envs/como/bin/python"
EVALUATOR="$SCRIPT_DIR/run_unet_direct_fullseq_greedy.py"
OUTPUT_DIR="$PROJECT_ROOT/channel_selection_results/step_j_unet_direct_fullseq_greedy"
CONFIG="$PROJECT_ROOT/como/config/como.yml"
CPU_AFFINITY="${CPU_AFFINITY:-0,1,4-9,12-15}"
NO_TURBO_PATH="/sys/devices/system/cpu/intel_pstate/no_turbo"

die() {
    printf '[ABORT] %s\n' "$*" >&2
    exit 1
}

for required in "$PYTHON" "$EVALUATOR" "$CONFIG"; do
    [[ -e "$required" ]] || die "Required path is missing: $required"
done

execute=0
for argument in "$@"; do
    [[ "$argument" == "--execute" ]] && execute=1
done

if (( execute )); then
    command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable"
    command -v taskset >/dev/null 2>&1 || die "taskset is unavailable"
    taskset -c "$CPU_AFFINITY" true 2>/dev/null || \
        die "Invalid or unavailable CPU affinity: $CPU_AFFINITY"
    [[ -r "$NO_TURBO_PATH" ]] || die "Cannot verify Intel Turbo state"
    [[ "$(<"$NO_TURBO_PATH")" == "1" ]] || \
        die "Intel Turbo is enabled; run: echo 1 | sudo tee $NO_TURBO_PATH"
    printf '[SAFETY] GPU: '
    nvidia-smi -L
    printf '[SAFETY] CPU affinity: %s; Intel Turbo: disabled\n' "$CPU_AFFINITY"
fi

mkdir -p "$OUTPUT_DIR/launch_backups"
if (( execute )); then
    stamp="$(date '+%Y%m%d_%H%M%S')"
    backup_dir="$OUTPUT_DIR/launch_backups/$stamp"
    mkdir -p "$backup_dir"
    cp -- "$CONFIG" "$backup_dir/como_before_launch.yml"
    database="$OUTPUT_DIR/evaluations.sqlite3"
    if [[ -f "$database" ]]; then
        "$PYTHON" - "$database" "$backup_dir/evaluations_before_launch.sqlite3" <<'PY'
import sqlite3
import sys

source_path, backup_path = sys.argv[1:]
source = sqlite3.connect(f"file:{source_path}?mode=ro", uri=True)
integrity = source.execute("PRAGMA integrity_check").fetchone()[0]
if integrity != "ok":
    raise SystemExit(f"[ABORT] SQLite integrity check failed: {integrity}")
target = sqlite3.connect(backup_path)
source.backup(target)
target.close()
source.close()
print(f"[BACKUP] Verified SQLite snapshot: {backup_path}")
PY
    fi
    printf '[BACKUP] Launch snapshot: %s\n' "$backup_dir"
fi

printf '%s\n' '=============================================================================='
printf '%s\n' 'UNET ENC1 DIRECT FULL-SEQUENCE GREEDY SEARCH'
printf '%s\n' '=============================================================================='
printf '[PLAN] Full fr1/desk_lightswitch only; maximum six selected channels\n'
printf '[PLAN] Primary ranking: keyframe evo_ape ATE mean with alignment and scale correction\n'
printf '[PLAN] K=4 is retained; adaptive final cardinality is selected from K=1--6 after repeats\n'
printf '[PLAN] Per-run timeout: 300 seconds unless overridden\n'
printf '[LAUNCH] Arguments:'
printf ' %q' "$@"
printf '\n\n'

set +e
taskset -c "$CPU_AFFINITY" "$PYTHON" "$EVALUATOR" "$@"
status=$?
set -e

if (( status != 0 )); then
    printf '[STOPPED] Evaluator exited with status %d. Saved SQLite rows remain resumable.\n' \
        "$status" >&2
    exit "$status"
fi

printf '[DONE] Results: %s\n' "$OUTPUT_DIR"
printf '[NEXT] Re-running this same command reuses completed candidate/replicate rows.\n'
