#!/bin/bash
#SBATCH --job-name=fw_greedy_bqs
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --output=/vol/bitbucket/mz325/individual_project/logs/forward_greedy_bqs_%j.out
#SBATCH --error=/vol/bitbucket/mz325/individual_project/logs/forward_greedy_bqs_%j.err

echo "=========================================="
echo "Forward Greedy Selection (BQS + Beam Search) — Channel Discovery"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "=========================================="

# ── Environment setup ──
eval "$(/vol/bitbucket/mz325/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate /vol/bitbucket/mz325/envs/como

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /vol/bitbucket/mz325/individual_project

if [ ! -f "forward_greedy_bqs.py" ]; then
    echo "Error: forward_greedy_bqs.py not found in $(pwd)"
    exit 1
fi

mkdir -p vis_results/forward_greedy_bqs
mkdir -p logs

echo ""
echo "--- Running Forward Greedy Selection (BQS + Beam Search) ---"
echo "  Objective : Basin Quality Score (LQBS, Width, Retention)"
echo "  Beam size : 3  (seeds: Top-3 single-channel BQS)"
echo "  Stop when : marginal BQS gain < 1e-4"
echo "  Max size  : no hard limit (early stopping controls)"
echo "  Frame     : 306"
echo "  Dataset   : freiburg1_desk"
echo ""

# ── Default run: beam_size=3, early stopping at min_improvement=1e-4 ──
python forward_greedy_bqs.py \
    --beam_size 3 \
    --min_improvement 1e-4 \
    --frame_indices 306

# ── Alternative: pure greedy (beam_size=1), useful for quick comparison ──
# python forward_greedy_bqs.py --beam_size 1 --min_improvement 1e-4 --frame_indices 306

# ── Alternative: multi-frame evaluation (slower, more robust) ──
# python forward_greedy_bqs.py --beam_size 3 --min_improvement 1e-4 --frame_indices 41 306 512

echo ""
echo "=========================================="
echo "Done at $(date)"
echo "Output: vis_results/forward_greedy_bqs/"
echo "=========================================="