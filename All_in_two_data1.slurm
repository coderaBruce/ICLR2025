#!/bin/bash
#SBATCH --job-name=ep5_d1
#SBATCH --account=PGS0218                
#SBATCH --nodes=1                      # 分配 x 个节点
#SBATCH --gpus-per-node=1             # 每个节点 y 个 GPU
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1           # 每节点跑一个进程
#SBATCH --partition=quad
#SBATCH --time=24:00:00
#SBATCH --output=logs/goat-%j.out

## data5的dora 和OPPU epoch5
# === 加载 Conda 环境 ===
module purge
module load cuda/12.4.1
module load miniconda3/24.1.2-py310
source activate goat

# === 进入你的项目路径 ===
cd /fs/ess/PGS0218/xli74/AAAI2025


# === 打印环境信息（调试用） ===
echo "====== ENVIRONMENT INFO ======"
echo "Hostname       : $(hostname)"
echo "Working dir    : $(pwd)"
echo "Date           : $(date)"
echo "Python         : $(which python)"
echo "CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"sq


# === 设置 Triton 本地 cache 避免挂起问题 ===
mkdir -p ./tmp/triton_cache
export TRITON_CACHE_DIR=./tmp/triton_cache
chmod +x All_in_two_data1.sh
srun bash All_in_two_data1.sh

