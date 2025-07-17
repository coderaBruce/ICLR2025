#!/bin/bash
#SBATCH --job-name=goat_try
#SBATCH --account=PGS0218                
#SBATCH --nodes=4                      # 分配 x 个节点
#SBATCH --gpus-per-node=2             # 每个节点 y 个 GPU
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1           # 每节点跑一个进程
#SBATCH --partition=gpu
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
echo "CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"

# === 运行你的 bash 脚本 ===
bash H100run_more_nodes_loop.sh
