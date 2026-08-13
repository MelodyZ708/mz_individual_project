#!/usr/bin/env bash
# Resumable full-sequence evaluation of the frozen 3,713-candidate second round.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
PYTHON="/home/melody/anaconda3/envs/como/bin/python"
EVO_APE="/home/melody/anaconda3/envs/como/bin/evo_ape"
EVALUATOR="$SCRIPT_DIR/run_full_sequence_evaluation.py"
OUTPUT_DIR="$PROJECT_ROOT/channel_selection_results/step_e_full_sequence_evaluation/second_round_baseline_plus2_rpe_safe"
PLAN="$OUTPUT_DIR/candidate_plan.json"
CONFIG="$PROJECT_ROOT/como/config/como.yml"
CPU_AFFINITY="${CPU_AFFINITY:-0,1,4-9,12-15}"
NO_TURBO_PATH="/sys/devices/system/cpu/intel_pstate/no_turbo"

die() {
    printf '[ABORT] %s\n' "$*" >&2
    exit 1
}

for required in "$PYTHON" "$EVO_APE" "$EVALUATOR" "$PLAN" "$CONFIG"; do
    [[ -e "$required" ]] || die "Required path is missing: $required"
done

"$PYTHON" - "$PLAN" <<'PY'
import json
import sys

path = sys.argv[1]
payload = json.load(open(path, encoding="utf-8"))
selection = payload.get("selection", {})
checks = {
    "protocol": payload.get("protocol") == "second_round_full_sequence_baseline_plus2_rpe_safe_v1",
    "selected_count": selection.get("selected_count") == 3713,
    "actual_candidates": len(payload.get("candidates", [])) == 3713,
    "associated_poses": selection.get("required_associated_poses") == 40,
    "ate_margin": selection.get("ate_margin_percent") == 2.0,
    "translation_rpe": selection.get("translation_rpe_max_cutoff_cm") == 6.0,
    "rotation_rpe": selection.get("rotation_rpe_max_cutoff_deg") == 5.0,
    "timeout": payload.get("timeout_seconds_per_run") == 300,
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit("[ABORT] Candidate-plan validation failed: " + ", ".join(failed))
print("[CHECK] Frozen candidate plan: 3,713 rows; thresholds and timeout verified")
PY

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable"
command -v journalctl >/dev/null 2>&1 || die "journalctl is unavailable"
command -v taskset >/dev/null 2>&1 || die "taskset is unavailable"
[[ -r "$NO_TURBO_PATH" ]] || die "Cannot verify Intel Turbo state"
[[ "$(<"$NO_TURBO_PATH")" == "1" ]] || \
    die "Intel Turbo is enabled; run: echo 1 | sudo tee $NO_TURBO_PATH"
taskset -c "$CPU_AFFINITY" true 2>/dev/null || \
    die "Invalid or unavailable CPU affinity: $CPU_AFFINITY"
if ! systemctl is-active --quiet rasdaemon.service; then
    die "rasdaemon.service is not active"
fi
if grep -Eq '(^| )(recovery|nomodeset|dis_ucode_ldr)( |$)' /proc/cmdline; then
    die "Unsafe recovery-mode boot detected; reboot normally"
fi

kernel_hardware_errors() {
    journalctl -b 0 -k --no-pager 2>/dev/null | grep -Ei \
        'mce: \[Hardware Error\]|machine check events logged|EDAC.*(error|uncorrected|corrected)|AER:.*error|NVRM: Xid|GPU has fallen off|out of memory|oom-kill' \
        || true
}

PREEXISTING_HARDWARE_ERRORS="$(kernel_hardware_errors)"
if [[ -n "$PREEXISTING_HARDWARE_ERRORS" ]]; then
    printf '%s\n' "$PREEXISTING_HARDWARE_ERRORS" >&2
    die "This boot already contains a hardware/driver error"
fi

printf '%s\n' '=============================================================================='
printf '%s\n' 'SECOND-ROUND FULL-SEQUENCE SEARCH: 3,713 MVS-QUALIFIED COMBINATIONS'
printf '%s\n' '=============================================================================='
printf '[CHECK] Primary rank: keyframe evo_ape ATE mean (--align --correct_scale)\n'
printf '[CHECK] Timeout per combination: 300 seconds\n'
printf '[CHECK] Overall wall-time limit: none; continue until all 3,713 finish\n'
printf '[CHECK] CPU affinity: %s; Intel Turbo disabled\n' "$CPU_AFFINITY"
printf '[CHECK] GPU: '
nvidia-smi -L || die "NVIDIA driver/GPU is not available"

mkdir -p "$OUTPUT_DIR/launch_backups"
STAMP="$(date '+%Y%m%d_%H%M%S')"
BACKUP_DIR="$OUTPUT_DIR/launch_backups/$STAMP"
mkdir -p "$BACKUP_DIR"
cp -- "$CONFIG" "$BACKUP_DIR/como_before_launch.yml"
cp -- "$PLAN" "$BACKUP_DIR/candidate_plan.json"

DB="$OUTPUT_DIR/evaluations.sqlite3"
if [[ -f "$DB" ]]; then
    "$PYTHON" - "$DB" "$BACKUP_DIR/evaluations_before_launch.sqlite3" <<'PY'
import sqlite3
import sys

source_path, backup_path = sys.argv[1:]
source = sqlite3.connect(f"file:{source_path}?mode=ro", uri=True)
integrity = source.execute("PRAGMA integrity_check").fetchone()[0]
if integrity != "ok":
    raise SystemExit(f"[ABORT] Source database integrity check failed: {integrity}")
backup = sqlite3.connect(backup_path)
source.backup(backup)
backup.close()
source.close()
print(f"[BACKUP] Verified SQLite snapshot: {backup_path}")
PY
fi

printf '[LAUNCH] Backup directory: %s\n' "$BACKUP_DIR"
printf '[LAUNCH] Extra evaluator arguments:'
printf ' %q' "$@"
printf '\n\n'

set +e
taskset -c "$CPU_AFFINITY" "$PYTHON" "$EVALUATOR" \
    --candidate-plan "$PLAN" \
    --output-dir "$OUTPUT_DIR" \
    --evo-ape "$EVO_APE" \
    --timeout-seconds 300 \
    "$@"
status=$?
set -e

POST_HARDWARE_ERRORS="$(kernel_hardware_errors)"
if [[ "$POST_HARDWARE_ERRORS" != "$PREEXISTING_HARDWARE_ERRORS" ]]; then
    printf '\n[HARDWARE WARNING] New current-boot event detected:\n%s\n' \
        "$POST_HARDWARE_ERRORS" >&2
    exit 3
fi
if (( status != 0 )); then
    printf '[STOPPED] Evaluator exited with status %d; saved rows remain resumable.\n' \
        "$status" >&2
    exit "$status"
fi

"$PYTHON" - "$DB" <<'PY'
import sqlite3
import sys

path = sys.argv[1]
connection = sqlite3.connect(path)
completed = connection.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
counts = dict(connection.execute("SELECT status, COUNT(*) FROM evaluations GROUP BY status"))
connection.close()
print(f"[BATCH RESULT] Preserved combinations: {completed:,}/3,713")
print(f"[BATCH RESULT] Remaining combinations: {3713 - completed:,}")
print("[BATCH RESULT] Status counts: " + ", ".join(f"{k}={v:,}" for k, v in sorted(counts.items())))
PY

printf '[DONE] No new matching kernel hardware event was detected.\n'
printf '[NEXT] Check: sudo ras-mc-ctl --errors\n'
printf '[NEXT] If interrupted, run this same command again to resume.\n'
