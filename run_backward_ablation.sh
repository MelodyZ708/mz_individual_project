#!/bin/bash
#SBATCH --job-name=bw_ablation
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/vol/bitbucket/mz325/individual_project/logs/backward_ablation_%j.out
#SBATCH --error=/vol/bitbucket/mz325/individual_project/logs/backward_ablation_%j.err

echo "=========================================="
echo "Backward Ablation — Channel Importance"
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

mkdir -p vis_results/backward_ablation
mkdir -p logs

echo ""
echo "--- Running Backward Ablation (Resume Mode) ---"
echo "  Top 2 combinations x 3 frames (early/mid/late)"
echo "  Skips completed combos, continues interrupted ones"
echo ""

# ── Option A: Resume from checkpoint ──
# Skips Rank01 (already complete), continues Rank02 from last checkpoint
python backward_ablation.py \
    --frame_indices 41 306 512 \
    --top_k 2 \
    --resume

# ── Option B: Fresh run (no resume) ──
# python backward_ablation.py --frame_indices 41 306 512 --top_k 2

# ── Option C: Use hardcoded fallback (if Step 1 hasn't been run yet) ──
# python backward_ablation.py --frame_indices 41 306 512 --top_k 2 --use_fallback --resume

# ── Option D: Test a single specific combination ──
# python backward_ablation.py --frame_indices 41 306 512 --channels "6,7,12,15,36,45,58,62"

echo ""
echo "=========================================="
echo "Done at $(date)"
echo "Output: vis_results/backward_ablation/"
echo "=========================================="