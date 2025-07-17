# === 设置 Triton 本地 cache 避免挂起问题 ===
mkdir -p ./tmp/triton_cache
export TRITON_CACHE_DIR=./tmp/triton_cache
chmod +x train_all_data1_goat_ddp.sh
srun bash train_all_data1_goat_ddp.sh