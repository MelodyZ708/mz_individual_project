#!/usr/bin/env bash
# Safely run an r=0.70 search stage with database backup, hardware checks,
# conservative CPU affinity, and atomic COMO configuration restoration.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
PYTHON="/home/melody/anaconda3/envs/como/bin/python"
EVALUATOR="$SCRIPT_DIR/run_r070_bruteforce.py"
OUTPUT_DIR="$PROJECT_ROOT/channel_selection_results/step_d_fail_fast_evaluation/r070_bruteforce_v2"
DB="$OUTPUT_DIR/evaluations.sqlite3"
MASTER_LOG="$OUTPUT_DIR/search_console.log"
CONFIG="$PROJECT_ROOT/como/config/como.yml"
LOCK="$OUTPUT_DIR/.como_config.lock"
EXPECTED_TOTAL=55554
MINIMUM_PRESERVED=40510
BATCH_HOURS="${BATCH_HOURS:-7}"
SEARCH_STAGE="${SEARCH_STAGE:-bruteforce}"
ALLOW_EXISTING_HARDWARE_EVENTS="${ALLOW_EXISTING_HARDWARE_EVENTS:-0}"
CPU_AFFINITY="${CPU_AFFINITY:-0,1,4-9,12-15}"
NO_TURBO_PATH="/sys/devices/system/cpu/intel_pstate/no_turbo"

die() {
    printf '[ABORT] %s\n' "$*" >&2
    exit 1
}

case "$SEARCH_STAGE" in
    bruteforce|retry-errors|swapback|rescue|repeat) ;;
    *) die "Unsupported SEARCH_STAGE=$SEARCH_STAGE" ;;
esac

for required in "$PYTHON" "$EVALUATOR" "$DB" "$CONFIG"; do
    [[ -e "$required" ]] || die "Required path is missing: $required"
done

command -v flock >/dev/null 2>&1 || die "flock is unavailable"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable"
command -v journalctl >/dev/null 2>&1 || die "journalctl is unavailable"
command -v taskset >/dev/null 2>&1 || die "taskset is unavailable"

[[ -r "$NO_TURBO_PATH" ]] || die "Cannot verify Intel Turbo state at $NO_TURBO_PATH"
[[ "$(<"$NO_TURBO_PATH")" == "1" ]] || \
    die "Intel Turbo is enabled; run: echo 1 | sudo tee $NO_TURBO_PATH"
taskset -c "$CPU_AFFINITY" true 2>/dev/null || \
    die "Invalid or unavailable CPU affinity: $CPU_AFFINITY"

if ! systemctl is-active --quiet rasdaemon.service; then
    die "rasdaemon.service is not active; hardware errors would not be recorded"
fi

kernel_hardware_errors() {
    journalctl -b 0 -k --no-pager 2>/dev/null | grep -Ei \
        'mce: \[Hardware Error\]|machine check events logged|EDAC.*(error|uncorrected|corrected)|AER:.*error|NVRM: Xid|GPU has fallen off|out of memory|oom-kill' \
        || true
}

PREEXISTING_HARDWARE_ERRORS="$(kernel_hardware_errors)"
if [[ -n "$PREEXISTING_HARDWARE_ERRORS" ]]; then
    if [[ "$ALLOW_EXISTING_HARDWARE_EVENTS" == "1" ]]; then
        [[ "$SEARCH_STAGE" == "bruteforce" ]] || \
            die "Hardware-event override is allowed only for SEARCH_STAGE=bruteforce"
        # This is deliberately a one-process environment override.  It is
        # limited to a short batch and is disabled again on the next normal
        # invocation.  Retain the baseline so the post-batch check can still
        # distinguish a newly added event from the already diagnosed event.
        if ! awk -v hours="$BATCH_HOURS" \
            'BEGIN { exit !(hours > 0 && hours <= 2) }'; then
            die "ALLOW_EXISTING_HARDWARE_EVENTS=1 is restricted to BATCH_HOURS <= 2"
        fi
        printf '%s\n' "$PREEXISTING_HARDWARE_ERRORS" >&2
        printf '%s\n' \
            '[OVERRIDE WARNING] Continuing for this process only despite the existing event.' \
            '[OVERRIDE WARNING] No existing results will be changed; the post-batch check remains active.' >&2
    else
        printf '%s\n' "$PREEXISTING_HARDWARE_ERRORS" >&2
        die "This boot already contains a hardware/driver error; do not start another batch"
    fi
fi

if grep -Eq '(^| )(recovery|nomodeset|dis_ucode_ldr)( |$)' /proc/cmdline; then
    die "Unsafe recovery-mode boot detected in /proc/cmdline; reboot normally before a long GPU search"
fi

# Refuse to start if another evaluator owns the shared COMO configuration.
exec 9>>"$LOCK"
flock -n 9 || die "Another channel-search process currently owns $LOCK"
flock -u 9
exec 9>&-

printf '%s\n' '=============================================================================='
printf '%s\n' 'R=0.70 SAFE SEARCH STAGE RUNNER'
printf '%s\n' '=============================================================================='
printf '[CHECK] Requested stage: %s\n' "$SEARCH_STAGE"
printf '[CHECK] Intel Turbo: disabled\n'
printf '[CHECK] Search CPU affinity: %s (logical CPUs 2,3,10,11 excluded)\n' "$CPU_AFFINITY"
printf '[CHECK] GPU availability: '
nvidia-smi -L || die "NVIDIA driver/GPU is not available"

STAMP="$(date '+%Y%m%d_%H%M%S')"
BACKUP_DIR="$OUTPUT_DIR/resume_backups/$STAMP"
mkdir -p "$BACKUP_DIR"

# SQLite's online backup API includes every committed WAL transaction.  The
# backup is then independently checked before anything else is changed.
"$PYTHON" - "$DB" "$BACKUP_DIR/evaluations_before_resume.sqlite3" \
    "$EXPECTED_TOTAL" "$MINIMUM_PRESERVED" <<'PY'
import sqlite3
import sys

source_path, backup_path, expected_total, minimum_preserved = sys.argv[1:]
expected_total = int(expected_total)
minimum_preserved = int(minimum_preserved)

source = sqlite3.connect(f"file:{source_path}?mode=ro", uri=True)
integrity = source.execute("PRAGMA integrity_check").fetchone()[0]
if integrity != "ok":
    raise SystemExit(f"[ABORT] Source database integrity check failed: {integrity}")

query = """
    SELECT COUNT(*), COUNT(DISTINCT candidate_key)
    FROM evaluations
    WHERE stage='bruteforce' AND replicate=0
"""
rows, unique_candidates = source.execute(query).fetchone()
if rows < minimum_preserved or unique_candidates < minimum_preserved:
    raise SystemExit(
        "[ABORT] Expected at least "
        f"{minimum_preserved:,} preserved candidates, found "
        f"rows={rows:,}, unique={unique_candidates:,}"
    )

backup = sqlite3.connect(backup_path)
source.backup(backup)
backup.close()
source.close()

check = sqlite3.connect(f"file:{backup_path}?mode=ro", uri=True)
backup_integrity = check.execute("PRAGMA integrity_check").fetchone()[0]
backup_rows, backup_unique = check.execute(query).fetchone()
check.close()
if backup_integrity != "ok" or (backup_rows, backup_unique) != (rows, unique_candidates):
    raise SystemExit(
        "[ABORT] Backup verification failed: "
        f"integrity={backup_integrity}, rows={backup_rows:,}, "
        f"unique={backup_unique:,}"
    )

remaining = max(0, expected_total - unique_candidates)
print(f"[CHECK] Source database integrity: ok")
print(f"[CHECK] Preserved brute-force results: {unique_candidates:,}/{expected_total:,}")
print(f"[CHECK] Remaining combinations: {remaining:,}")
print(f"[BACKUP] Verified SQLite snapshot: {backup_path}")
PY

cp -- "$CONFIG" "$BACKUP_DIR/como_config_found_after_crash.yml"
if [[ "$SEARCH_STAGE" == "retry-errors" && -d "$OUTPUT_DIR/artifacts/bruteforce" ]]; then
    cp -a -- "$OUTPUT_DIR/artifacts/bruteforce" \
        "$BACKUP_DIR/bruteforce_error_artifacts_before_retry"
    printf '[BACKUP] Preserved pre-retry brute-force error artifacts: %s\n' \
        "$BACKUP_DIR/bruteforce_error_artifacts_before_retry"
fi
if [[ -f "$MASTER_LOG" ]]; then
    cp -- "$MASTER_LOG" "$BACKUP_DIR/search_console_before_resume.log"

    # The crash left only NUL bytes after the last complete log line.  Preserve
    # the byte-for-byte copy above, then remove that corrupt tail so subsequent
    # appended output remains a normal text file.
    "$PYTHON" - "$MASTER_LOG" <<'PY'
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = path.read_bytes()
clean = payload.rstrip(b"\x00")
if len(clean) != len(payload):
    temporary = path.with_name(f".{path.name}.resume-clean.tmp")
    temporary.write_bytes(clean)
    os.replace(temporary, path)
    print(f"[REPAIR] Removed {len(payload) - len(clean):,} trailing NUL bytes from {path}")
else:
    print("[CHECK] Master log has no trailing NUL bytes")
PY
fi

# The hard shutdown occurred after the evaluator applied [26,29,30,50], so its
# finally block could not restore the pre-search state.  Restore only the keys
# controlled by this evaluation, preserving all unrelated COMO settings.
"$PYTHON" - "$CONFIG" <<'PY'
import os
import sys
import tempfile
from pathlib import Path

import yaml

path = Path(sys.argv[1])
config = yaml.safe_load(path.read_text(encoding="utf-8"))
tracking = config["tracking"]
tracking["cnn_channel_select"] = "d5,d29,d40,d52"
tracking["debug_tracking_diagnostics"] = False
tracking.pop("debug_tracking_print_every_frame", None)
tracking.pop("debug_tracking_save_suspicious", None)

payload = yaml.safe_dump(
    config, default_flow_style=False, allow_unicode=True, sort_keys=False
).encode("utf-8")
descriptor, temporary_name = tempfile.mkstemp(
    prefix=f".{path.name}.", suffix=".resume.tmp", dir=path.parent
)
try:
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_name, path)
finally:
    if os.path.exists(temporary_name):
        os.unlink(temporary_name)
print("[CONFIG] Restored pre-search baseline channels [5,29,40,52]")
PY

cp -- "$CONFIG" "$BACKUP_DIR/como_baseline_for_resume.yml"

restore_baseline() {
    if [[ -f "$BACKUP_DIR/como_baseline_for_resume.yml" ]]; then
        cp -- "$BACKUP_DIR/como_baseline_for_resume.yml" "$CONFIG"
    fi
}
trap restore_baseline EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

if [[ "$SEARCH_STAGE" == "bruteforce" ]]; then
    printf '[RESUME] Existing database rows will be skipped; --rerun-existing is not used.\n'
    printf '[RESUME] Launching a %.2f-hour brute-force batch.\n' "$BATCH_HOURS"
elif [[ "$SEARCH_STAGE" == "retry-errors" ]]; then
    printf '[RETRY] Only retryable brute-force errors will be replaced.\n'
else
    printf '[SEARCH] Existing exact stage results will be skipped and preserved.\n'
fi
printf '[RESUME] Backup directory: %s\n\n' "$BACKUP_DIR"

EVALUATOR_ARGS=(--stage "$SEARCH_STAGE" --execute)
if [[ "$SEARCH_STAGE" == "bruteforce" ]]; then
    EVALUATOR_ARGS+=(--max-stage-hours "$BATCH_HOURS")
fi

set +e
taskset -c "$CPU_AFFINITY" "$PYTHON" "$EVALUATOR" "${EVALUATOR_ARGS[@]}"
status=$?
set -e

if (( status != 0 )); then
    printf '\n[STOPPED] Evaluator exited with status %d; preserved database rows remain resumable.\n' "$status" >&2
    exit "$status"
fi

POST_BATCH_HARDWARE_ERRORS="$(kernel_hardware_errors)"
if [[ "$POST_BATCH_HARDWARE_ERRORS" != "$PREEXISTING_HARDWARE_ERRORS" ]]; then
    printf '\n[HARDWARE WARNING] The current-boot hardware-event set changed during this batch:\n%s\n' \
        "$POST_BATCH_HARDWARE_ERRORS" >&2
    printf '[STOP] Do not start another batch until these events are diagnosed.\n' >&2
    exit 3
fi

if [[ -n "$PREEXISTING_HARDWARE_ERRORS" ]]; then
    printf '\n[HARDWARE CHECK] No additional matching kernel event appeared during this batch.\n'
    printf '[HARDWARE CHECK] The pre-existing event remains; the next normal invocation will block again.\n'
fi

"$PYTHON" - "$DB" "$EXPECTED_TOTAL" <<'PY'
import sqlite3
import sys

database, expected_total = sys.argv[1], int(sys.argv[2])
connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
completed = connection.execute(
    "SELECT COUNT(DISTINCT candidate_key) FROM evaluations "
    "WHERE stage='bruteforce' AND replicate=0"
).fetchone()[0]
connection.close()
remaining = max(0, expected_total - completed)
print(f"[BATCH RESULT] Preserved combinations: {completed:,}/{expected_total:,}")
print(f"[BATCH RESULT] Remaining combinations: {remaining:,}")
if remaining:
    print("[NEXT] Run: sudo ras-mc-ctl --errors")
    print("[NEXT] If every category reports no errors, launch this same script again.")
else:
    print("[DONE] All brute-force combinations are complete.")
PY
