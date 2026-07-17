#!/bin/bash
# ============================================================
# Random Channel Search — layer1 (500 trials) then layer2 (1500 trials)
# fr1/desk lightswitch, results stored separately
# Channel count cycles: 6,6,6,6, 4,4,4,4, 8,8,8,8, repeat
# Timeout: 300s per trial → FAIL
# ============================================================

PROJECT_DIR="/home/melody/code/individual_project/como"
CONFIG_FILE="${PROJECT_DIR}/config/como.yml"
CONFIG_BACKUP="${PROJECT_DIR}/config/como.yml.bak"
TRAJ_FILE="${PROJECT_DIR}/results/data_tum.txt"
DATASET_DIR="/home/melody/data/tum/rgbd_dataset_freiburg1_desk_lightswitch"
GT_FILE="${DATASET_DIR}/groundtruth.txt"

# ── Backup config once, restore on exit ───────────────────────────────────────
cp "${CONFIG_FILE}" "${CONFIG_BACKUP}"
trap "echo '[Cleanup] Restoring config...'; cp '${CONFIG_BACKUP}' '${CONFIG_FILE}'" EXIT
cd "${PROJECT_DIR}"

# ==============================================================================
# Helper: run one search phase
#   $1 = layer name (layer1 / layer2)
#   $2 = full_channels (64 / 128)
#   $3 = num_trials
#   $4 = results_dir
# ==============================================================================
run_search() {
    local LAYER_NAME="$1"
    local FULL_CHANNELS="$2"
    local NUM_TRIALS="$3"
    local RESULTS_DIR="$4"
    local SUMMARY_FILE="${RESULTS_DIR}/summary.tsv"

    echo ""
    echo "##########################################################"
    echo "# Starting: ${LAYER_NAME}  |  ${NUM_TRIALS} trials"
    echo "# Results → ${RESULTS_DIR}"
    echo "# Host: $(hostname) | Date: $(date)"
    echo "##########################################################"

    mkdir -p "${RESULTS_DIR}"

    # ── Generate trial plan ──────────────────────────────────────────────────
    python3 - <<PYEOF
import random, json, os

random.seed(42)

pattern = [6]*4 + [4]*4 + [8]*4   # 12-cycle
num_trials = ${NUM_TRIALS}
full_channels = ${FULL_CHANNELS}
trials = []
for i in range(num_trials):
    n_ch = pattern[i % len(pattern)]
    channels = sorted(random.sample(range(full_channels), n_ch))
    ch_str = ",".join(f"d{c}" for c in channels)
    trials.append({"trial": i+1, "n_ch": n_ch, "channels": ch_str})

out_path = "${RESULTS_DIR}/trial_plan.json"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(trials, f, indent=2)
print(f"[Setup] Generated {num_trials} trial specs → {out_path}")
PYEOF

    # ── Write TSV header ─────────────────────────────────────────────────────
    echo -e "Trial\tN_Channels\tChannels\tATE_Mean_cm\tStatus" > "${SUMMARY_FILE}"

    # ── Main loop ────────────────────────────────────────────────────────────
    python3 - <<PYEOF
import json, subprocess, os, shutil
import yaml

trial_plan   = json.load(open("${RESULTS_DIR}/trial_plan.json"))
config_file  = "${CONFIG_FILE}"
traj_file    = "${TRAJ_FILE}"
results_dir  = "${RESULTS_DIR}"
gt_file      = "${GT_FILE}"
dataset_dir  = "${DATASET_DIR}"
summary_file = "${SUMMARY_FILE}"
project_dir  = "${PROJECT_DIR}"
layer_name   = "${LAYER_NAME}"
full_channels = ${FULL_CHANNELS}

MIN_FRAMES    = 400
FROZEN_WINDOW = 30
FROZEN_UNIQUE = 3

pad = 4 if len(trial_plan) > 999 else 3


def apply_config(ch_str, n_ch):
    cfg = yaml.safe_load(open(config_file))
    cfg["tracking"]["color"]                   = "cnn"
    cfg["tracking"]["cnn_mode"]                = "cnn_only"
    cfg["tracking"]["cnn_layer_name"]          = layer_name
    cfg["tracking"]["cnn_channels"]            = n_ch
    cfg["tracking"]["cnn_channel_select"]      = ch_str
    cfg["tracking"]["cnn_layer_full_channels"] = full_channels
    yaml.dump(cfg, open(config_file, "w"), default_flow_style=False, allow_unicode=True)


def run_como():
    try:
        subprocess.run(
            ["python", "como/como_dataset.py",
             "--dataset_type=tum",
             f"--dataset_dir={dataset_dir}"],
            cwd=project_dir,
            timeout=300,
            capture_output=False,
            env=os.environ.copy()
        )
        return True
    except subprocess.TimeoutExpired:
        return False


def compute_ate(traj_saved):
    try:
        out = subprocess.check_output(
            ["evo_ape", "tum", gt_file, traj_saved,
             "--align", "--correct_scale"],
            stderr=subprocess.DEVNULL
        ).decode()
        for line in out.splitlines():
            if "mean" in line.lower() and line.strip().split()[0] == "mean":
                return f"{float(line.strip().split()[1])*100:.4f}"
    except Exception:
        pass
    return None


def has_nan(path):
    with open(path) as f:
        return "nan" in f.read().lower()


def check_traj_valid(path):
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    if len(lines) < MIN_FRAMES:
        return False, f"too_short ({len(lines)} frames < {MIN_FRAMES})"
    tail_poses = []
    for line in lines[-FROZEN_WINDOW:]:
        parts = line.split()
        if len(parts) >= 4:
            tail_poses.append((round(float(parts[1]), 4),
                               round(float(parts[2]), 4),
                               round(float(parts[3]), 4)))
    unique_poses = len(set(tail_poses))
    if unique_poses <= FROZEN_UNIQUE:
        return False, f"frozen (only {unique_poses} unique poses in last {FROZEN_WINDOW} frames)"
    return True, "ok"


for t in trial_plan:
    trial_id   = t["trial"]
    n_ch       = t["n_ch"]
    ch_str     = t["channels"]
    traj_saved = os.path.join(results_dir, f"trial_{trial_id:0{pad}d}.txt")

    print(f"\n{'#'*50}")
    print(f"# [{layer_name}] Trial {trial_id:{pad}d}/{len(trial_plan)}  |  n_ch={n_ch}  |  {ch_str}")
    print(f"{'#'*50}")

    if os.path.exists(traj_file):
        os.remove(traj_file)

    apply_config(ch_str, n_ch)
    ok = run_como()

    if not ok:
        print("  [FAIL] Timeout (>300s)")
        with open(summary_file, "a") as f:
            f.write(f"{trial_id}\t{n_ch}\t{ch_str}\t-\tTIMEOUT\n")
        continue

    if not os.path.exists(traj_file):
        print("  [FAIL] Trajectory not produced")
        with open(summary_file, "a") as f:
            f.write(f"{trial_id}\t{n_ch}\t{ch_str}\t-\tFAIL\n")
        continue

    if has_nan(traj_file):
        print("  [FAIL] Trajectory contains NaN")
        shutil.copy(traj_file, traj_saved + ".nan")
        with open(summary_file, "a") as f:
            f.write(f"{trial_id}\t{n_ch}\t{ch_str}\t-\tNAN_FAIL\n")
        continue

    valid, reason = check_traj_valid(traj_file)
    if not valid:
        print(f"  [FAIL] Trajectory invalid: {reason}")
        shutil.copy(traj_file, traj_saved + ".frozen")
        with open(summary_file, "a") as f:
            f.write(f"{trial_id}\t{n_ch}\t{ch_str}\t-\tFROZEN_FAIL\n")
        continue

    shutil.copy(traj_file, traj_saved)
    ate = compute_ate(traj_saved)

    if ate is None:
        print("  [FAIL] ATE parse error")
        with open(summary_file, "a") as f:
            f.write(f"{trial_id}\t{n_ch}\t{ch_str}\t-\tPARSE_FAIL\n")
    else:
        print(f"  ATE Mean = {ate} cm  |  OK")
        with open(summary_file, "a") as f:
            f.write(f"{trial_id}\t{n_ch}\t{ch_str}\t{ate}\tOK\n")

print("\n==========================================")
print(f"[{layer_name}] All trials complete.")
print("==========================================")
PYEOF

    # ── Print top-20 for this phase ──────────────────────────────────────────
    echo ""
    echo "=========================================="
    echo "TOP 20 — ${LAYER_NAME} (by ATE Mean, lower is better)"
    echo "=========================================="
    printf "%-8s  %-6s  %-40s  %14s\n" "Trial" "N_ch" "Channels" "ATE Mean (cm)"
    printf "%-8s  %-6s  %-40s  %14s\n" "--------" "------" "----------------------------------------" "--------------"
    grep "OK$" "${SUMMARY_FILE}" | sort -t$'\t' -k4 -n | head -20 | \
        while IFS=$'\t' read -r trial n_ch ch ate stat; do
            printf "%-8s  %-6s  %-40s  %14s\n" "${trial}" "${n_ch}" "${ch}" "${ate}"
        done
    echo ""
    echo "Full results: ${SUMMARY_FILE}"
    echo "Phase ${LAYER_NAME} done at $(date)"
}

# ==============================================================================
# Run layer1 first, then layer2
# ==============================================================================
run_search "layer1" 64  500  "${PROJECT_DIR}/results/random_channel_search_layer1"
run_search "layer2" 128 1500 "${PROJECT_DIR}/results/random_channel_search_layer2"

echo ""
echo "=========================================="
echo "All phases complete at $(date)"
echo "=========================================="