#!/bin/bash
#SBATCH --job-name=greedy_all
#SBATCH --output=logs/greedy_all_%j.out
#SBATCH --error=logs/greedy_all_%j.err
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

echo "=========================================="
echo "Unified Forward Greedy BQS Search"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"
echo "=========================================="

eval "$(/vol/bitbucket/mz325/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate /vol/bitbucket/mz325/envs/como

cd /vol/bitbucket/mz325/individual_project

mkdir -p logs
mkdir -p vis_results

# 定义要测试的配置
LAYERS=("conv1" "layer1" "layer2")
RELU_MODES=("pre" "post")

for layer in "${LAYERS[@]}"; do
    for relu in "${RELU_MODES[@]}"; do
        
        echo ""
        echo "---------------------------------------------------"
        echo "Starting: Layer = $layer, ReLU Mode = $relu"
        echo "---------------------------------------------------"
        
        # 运行 Python 脚本
        python forward_greedy_bqs_unified.py \
            --layer $layer \
            --relu_mode $relu \
            --beam_size 3 \
            --min_improvement 1e-4 \
            --max_channels 64
            
        echo "Finished: Layer = $layer, ReLU Mode = $relu"
        
    done
done

echo ""
echo "=========================================="
echo "All configurations completed."
echo "Date: $(date)"
echo "=========================================="