#!/bin/bash
#SBATCH --job-name=fw_greedy_v3
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/vol/bitbucket/mz325/individual_project/logs/forward_greedy_%j.out
#SBATCH --error=/vol/bitbucket/mz325/individual_project/logs/forward_greedy_%j.err

echo "=========================================="
echo "Forward Greedy Selection v3 — Channel Discovery"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "=========================================="

# ── Environment setup ──
eval "$(/vol/bitbucket/mz325/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate /vol/bitbucket/mz325/envs/como

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

cd /vol/bitbucket/mz325/individual_project

mkdir -p vis_results/forward_greedy_selection
mkdir -p logs

echo ""
echo "--- Running Forward Greedy Selection v3 ---"
echo "  Phase 1: Rank01 internal (exhaustive if alive<=5, else greedy)"
echo "  Phase 2: Rank02 internal (greedy + deduped look-ahead)"
echo "  Phase 3: Cross-combination (greedy from P1+P2 best seeds)"
echo "  Objective: 0.7*Sharp_50 + 0.3*Sharp_Clean"
echo "  Frames: 41, 306, 512"
echo ""

# ── Option A: Resume from v2 checkpoint (reuse completed phases + single rankings) ──
python forward_greedy_selection.py \
    --frame_indices 41 306 512 \
    --resume

# ── Option B: Fresh run (no resume, re-evaluate everything) ──
# python forward_greedy_selection.py --frame_indices 41 306 512

# ── Option C: Skip Phase 3 (only run internal selections) ──
# python forward_greedy_selection.py --frame_indices 41 306 512 --resume --skip_phase3

# ── Option D: Custom alpha weight ──
# python forward_greedy_selection.py --frame_indices 41 306 512 --alpha 0.6 --resume

echo ""
echo "=========================================="
echo "Done at $(date)"
echo "Output: vis_results/forward_greedy_selection/"
echo "=========================================="