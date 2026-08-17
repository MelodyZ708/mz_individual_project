#!/usr/bin/env bash
# Resumable U-Net Enc0/Enc1 evaluation: 3 TUM families × 3 lighting conditions.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
PYTHON="/home/melody/anaconda3/envs/como/bin/python"
EVALUATOR="$SCRIPT_DIR/run_unet_multi_dataset_evaluation.py"
AGGREGATOR="$SCRIPT_DIR/aggregate_unet_multi_dataset_results.py"
DATASET_PLAN="$SCRIPT_DIR/unet_dataset_plan.json"
CANDIDATE_PLAN="$SCRIPT_DIR/unet_candidate_plan.json"
OUTPUT_DIR="$PROJECT_ROOT/channel_selection_results/step_m_unet_multi_dataset_evaluation"
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
  run_unet_multi_dataset_evaluation.sh             # validate only; does not launch COMO
  run_unet_multi_dataset_evaluation.sh --execute   # run/resume 115 active evaluations
  run_unet_multi_dataset_evaluation.sh --aggregate-only

The --execute command is resumable. One SQLite database is kept per dataset;
completed label/replicate rows are skipped automatically on re-run.
EOF
}

mode="validate"
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

for required in "$PYTHON" "$EVALUATOR" "$AGGREGATOR" "$DATASET_PLAN" \
    "$CANDIDATE_PLAN" "$CONFIG"; do
    [[ -e "$required" ]] || die "Required path is missing: $required"
done

mapfile -t DATASET_ROWS < <(
    "$PYTHON" - "$DATASET_PLAN" "$CANDIDATE_PLAN" <<'PY'
import json
import pathlib
import sys

dataset_plan_path, candidate_plan_path = map(pathlib.Path, sys.argv[1:])
datasets_doc = json.loads(dataset_plan_path.read_text(encoding="utf-8"))
candidates_doc = json.loads(candidate_plan_path.read_text(encoding="utf-8"))
if datasets_doc.get("protocol") != "unet_enc0_enc1_three_dataset_three_lighting_conditions_v1":
    raise SystemExit("[ABORT] Unexpected U-Net dataset protocol")
if candidates_doc.get("protocol") != "unet_enc0_enc1_multi_dataset_candidate_plan_v1":
    raise SystemExit("[ABORT] Unexpected U-Net candidate protocol")
datasets = datasets_doc.get("datasets", [])
candidates = candidates_doc.get("candidates", [])
if len(datasets) != 9 or datasets_doc.get("dataset_count") != 9:
    raise SystemExit("[ABORT] Expected exactly nine datasets")
if len(candidates) != 13 or candidates_doc.get("selection", {}).get("selected_count") != 13:
    raise SystemExit("[ABORT] Expected exactly thirteen candidates")
if datasets_doc.get("timeout_seconds_per_run") != 500 or datasets_doc.get("replicates_per_candidate") != 1:
    raise SystemExit("[ABORT] Expected exactly one replicate and 500-second timeout")
if {item.get("enc_level") for item in candidates} != {0, 1}:
    raise SystemExit("[ABORT] Candidate plan must contain Enc0 and Enc1")
if len({item.get("label") for item in candidates}) != len(candidates):
    raise SystemExit("[ABORT] Candidate labels are not unique")
if len({item.get("candidate_key") for item in candidates}) != len(candidates):
    raise SystemExit("[ABORT] Candidate keys are not unique")
candidate_keys = {item["candidate_key"] for item in candidates}
root = pathlib.Path(datasets_doc["dataset_root"])
for order, item in enumerate(datasets, start=1):
    if item.get("order") != order:
        raise SystemExit("[ABORT] Dataset order is not frozen")
    dataset = root / item["directory_name"]
    for name in ("matched_rgb.txt", "matched_depth.txt", "groundtruth.txt"):
        if not (dataset / name).is_file():
            raise SystemExit(f"[ABORT] Missing {dataset / name}")
    with (dataset / "matched_rgb.txt").open(encoding="utf-8") as handle:
        frame_count = sum(1 for line in handle if line.strip() and not line.lstrip().startswith("#"))
    if frame_count != item["expected_matched_frames"]:
        raise SystemExit(
            f"[ABORT] {item['key']} matched frame count changed: "
            f"{frame_count} != {item['expected_matched_frames']}"
        )
    excluded = item.get("excluded_candidate_keys", [])
    if not isinstance(excluded, list) or len(excluded) != len(set(excluded)):
        raise SystemExit(f"[ABORT] Invalid/duplicate exclusions for {item['key']}")
    if set(excluded) - candidate_keys:
        raise SystemExit(f"[ABORT] Unknown exclusions for {item['key']}: {set(excluded) - candidate_keys}")
    reason = str(item.get("exclusion_reason", "")).strip()
    if excluded and not reason:
        raise SystemExit(f"[ABORT] Missing exclusion reason for {item['key']}")
    print("\t".join((
        item["key"], str(dataset), str(frame_count), item["family"], item["condition"],
        ",".join(excluded), reason,
    )))
PY
)

(( ${#DATASET_ROWS[@]} == 9 )) || die "Dataset-plan validation returned ${#DATASET_ROWS[@]} rows"

printf '%s\n' '=============================================================================='
printf '%s\n' 'U-NET ENC0/ENC1: THREE DATASETS × THREE LIGHTING CONDITIONS'
printf '%s\n' '=============================================================================='
printf '[MODE] %s\n' "$mode"
printf '[PLAN] 9 sequences × 13 configurations = 117 nominal cells; 2 fr3-lightswitch all-channel controls are safety-excluded = 115 active runs\n'
printf '[PLAN] Tracking: U-Net Enc0/Enc1; mapping: gray + sensor depth\n'
printf '[PLAN] Primary metric: keyframe evo_ape ATE mean (--align --correct_scale)\n'
printf '[PLAN] Diagnostics: historical RPE, all-frame SE(3) ATE/RPE, coverage, numerical fields\n'
printf '[PLAN] Timeout per run: %d seconds\n' "$TIMEOUT_SECONDS"
printf '[OUTPUT] %s\n' "$OUTPUT_DIR"
for row in "${DATASET_ROWS[@]}"; do
    IFS=$'\t' read -r key path frames family condition excluded reason <<< "$row"
    printf '  [%s] frames=%s family=%s condition=%s' "$key" "$frames" "$family" "$condition"
    [[ -n "$excluded" ]] && printf ' safety-excluded=%s' "$excluded"
    printf '\n'
done

if [[ "$mode" == "aggregate" ]]; then
    "$PYTHON" "$AGGREGATOR" \
        --output-dir "$OUTPUT_DIR" \
        --dataset-plan "$DATASET_PLAN" \
        --candidate-plan "$CANDIDATE_PLAN"
    exit 0
fi

if [[ "$mode" == "validate" ]]; then
    for row in "${DATASET_ROWS[@]}"; do
        IFS=$'\t' read -r key path _frames _family _condition excluded reason <<< "$row"
        exclusion_args=()
        if [[ -n "$excluded" ]]; then
            IFS=',' read -r -a excluded_keys <<< "$excluded"
            for candidate_key in "${excluded_keys[@]}"; do
                exclusion_args+=(--exclude-candidate-key "$candidate_key")
            done
            exclusion_args+=(--exclusion-reason "$reason")
        fi
        "$PYTHON" "$EVALUATOR" \
            --validate-only \
            --dataset-dir "$path" \
            --candidate-plan "$CANDIDATE_PLAN" \
            --timeout-seconds "$TIMEOUT_SECONDS" \
            "${exclusion_args[@]}"
        printf '[VALID] %s\n' "$key"
    done
    printf '\n[VALIDATION COMPLETE] No COMO process was launched and no result database was written.\n'
    printf '[RUN] %s --execute\n' "$0"
    exit 0
fi

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable"
command -v taskset >/dev/null 2>&1 || die "taskset is unavailable"
taskset -c "$CPU_AFFINITY" true 2>/dev/null || die "Invalid CPU affinity: $CPU_AFFINITY"
[[ -r "$NO_TURBO_PATH" ]] || die "Cannot verify Intel Turbo state"
[[ "$(<"$NO_TURBO_PATH")" == "1" ]] || \
    die "Intel Turbo is enabled; run: echo 1 | sudo tee $NO_TURBO_PATH"
printf '[SAFETY] GPU: '
nvidia-smi -L || die "NVIDIA driver/GPU is unavailable"
printf '[SAFETY] CPU affinity=%s; Intel Turbo=disabled\n' "$CPU_AFFINITY"

mkdir -p "$OUTPUT_DIR/per_dataset" "$OUTPUT_DIR/launch_backups"
stamp="$(date '+%Y%m%d_%H%M%S')"
backup_dir="$OUTPUT_DIR/launch_backups/$stamp"
mkdir -p "$backup_dir/databases"
cp -- "$CONFIG" "$backup_dir/como_before_launch.yml"
cp -- "$DATASET_PLAN" "$backup_dir/unet_dataset_plan.json"
cp -- "$CANDIDATE_PLAN" "$backup_dir/unet_candidate_plan.json"

restore_config() {
    if [[ -f "$backup_dir/como_before_launch.yml" ]]; then
        cp -- "$backup_dir/como_before_launch.yml" "$CONFIG"
    fi
}
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
    raise SystemExit(f"[ABORT] SQLite integrity check failed: {source_path}")
target = sqlite3.connect(backup_path)
source.backup(target)
target.close()
source.close()
print(f"[BACKUP] SQLite snapshot: {backup_path}")
PY
    fi
done
printf '[BACKUP] Launch snapshot: %s\n' "$backup_dir"

for row in "${DATASET_ROWS[@]}"; do
    IFS=$'\t' read -r key path frames family condition excluded reason <<< "$row"
    dataset_output="$OUTPUT_DIR/per_dataset/$key"
    printf '\n[DATASET START] %s (%s frames; %s/%s)\n' "$key" "$frames" "$family" "$condition"
    exclusion_args=()
    if [[ -n "$excluded" ]]; then
        IFS=',' read -r -a excluded_keys <<< "$excluded"
        for candidate_key in "${excluded_keys[@]}"; do
            exclusion_args+=(--exclude-candidate-key "$candidate_key")
        done
        exclusion_args+=(--exclusion-reason "$reason")
        printf '[SAFETY] Excluding %s\n' "$excluded"
    fi
    taskset -c "$CPU_AFFINITY" "$PYTHON" "$EVALUATOR" \
        --execute \
        --dataset-dir "$path" \
        --output-dir "$dataset_output" \
        --candidate-plan "$CANDIDATE_PLAN" \
        --timeout-seconds "$TIMEOUT_SECONDS" \
        "${exclusion_args[@]}"
    "$PYTHON" "$AGGREGATOR" \
        --output-dir "$OUTPUT_DIR" \
        --dataset-plan "$DATASET_PLAN" \
        --candidate-plan "$CANDIDATE_PLAN"
done

"$PYTHON" "$AGGREGATOR" \
    --output-dir "$OUTPUT_DIR" \
    --dataset-plan "$DATASET_PLAN" \
    --candidate-plan "$CANDIDATE_PLAN"
printf '\n[DONE] All 115 active U-Net multi-dataset cells were processed; 2 fr3-lightswitch all-channel controls were safety-excluded.\n'
printf '[READ FIRST] %s/aggregate_summary.md\n' "$OUTPUT_DIR"
printf '[TABLE] %s/dataset_scorecard.csv\n' "$OUTPUT_DIR"
