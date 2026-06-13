#!/bin/bash
#SBATCH --job-name=como_cnn_5runs
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=05:00:00
#SBATCH --output=/vol/bitbucket/mz325/individual_project/logs/cnn_5runs_%j.out
#SBATCH --error=/vol/bitbucket/mz325/individual_project/logs/cnn_5runs_%j.err

echo "=========================================="
echo "CoMo CNN 5-Run ATE Evaluation (Affine Fixed)"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "=========================================="

eval "$(/vol/bitbucket/mz325/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate /vol/bitbucket/mz325/envs/como

cd /vol/bitbucket/mz325/individual_project/como
mkdir -p results

RMSE_SUM=0
MEAN_SUM=0
VALID_RUNS=0

for RUN in 1 2 3 4 5; do
    echo ""
    echo "========== RUN $RUN / 5 =========="

    Xvfb :$((200 + RUN)) -screen 0 1920x1080x24 &
    XVFB_PID=$!
    sleep 1
    export DISPLAY=:$((200 + RUN))

    timeout 900 python como/como_dataset.py \
        --dataset_type=tum \
        --dataset_dir=/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/ || true

    kill $XVFB_PID 2>/dev/null

    TRAJ="results/tum_rgbd_dataset_freiburg1_desk.txt"
    if [ -f "$TRAJ" ]; then
        cp "$TRAJ" "results/fr1desk_cnn_affine_fixed_run${RUN}.txt"
        echo "--- ATE for Run $RUN ---"
        RESULT=$(evo_ape tum \
            /vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/groundtruth.txt \
            "results/fr1desk_cnn_affine_fixed_run${RUN}.txt" \
            --align --correct_scale 2>&1)
        echo "$RESULT"

        RMSE=$(echo "$RESULT" | grep "rmse" | awk '{print $2}')
        MEAN=$(echo "$RESULT" | grep "mean" | awk '{print $2}')

        if [ ! -z "$RMSE" ]; then
            RMSE_SUM=$(python3 -c "print($RMSE_SUM + $RMSE)")
            MEAN_SUM=$(python3 -c "print($MEAN_SUM + $MEAN)")
            VALID_RUNS=$((VALID_RUNS + 1))
            echo "Run $RUN: RMSE=$RMSE m, Mean=$MEAN m"
        else
            echo "Run $RUN: ATE computation failed (degenerate trajectory?)"
        fi
    else
        echo "Run $RUN: Trajectory file not found"
    fi
done

echo ""
echo "=========================================="
echo "SUMMARY ($VALID_RUNS valid runs)"
echo "=========================================="
if [ $VALID_RUNS -gt 0 ]; then
    python3 -c "
rmse_sum = $RMSE_SUM
mean_sum = $MEAN_SUM
n = $VALID_RUNS
print(f'Avg RMSE: {rmse_sum/n*100:.3f} cm')
print(f'Avg Mean: {mean_sum/n*100:.3f} cm')
"
fi

echo "All done at $(date)"
