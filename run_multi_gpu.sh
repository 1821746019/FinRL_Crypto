#!/bin/bash

# 检查CUDA是否可用
echo "正在检查GPU是否可用..."
if [ -x "$(command -v nvidia-smi)" ]; then
    nvidia-smi
    num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "检测到${num_gpus}个GPU"
    
    if [ "$num_gpus" -gt 1 ]; then
        # 多GPU设置
        all_gpus=$(seq -s, 0 $((num_gpus-1)))
        echo "将使用所有GPU: $all_gpus"
        echo "启动多GPU训练..."
        
        # 使用SAC算法，在所有GPU上训练
        python 1_optimize_cpcv.py --gpus=$all_gpus --model=sac --name=multi_gpu_model
    else
        # 单GPU设置
        echo "只有1个GPU，将使用单GPU训练模式..."
        python 1_optimize_cpcv.py --gpus=0 --model=sac --name=single_gpu_model
    fi
else
    echo "未检测到GPU，将使用CPU训练..."
    python 1_optimize_cpcv.py --gpus=-1 --model=sac --name=cpu_model
fi 