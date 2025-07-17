# === 设置 Triton 本地 cache 避免挂起问题 ===
export TRITON_CACHE_DIR=./tmp/triton_cache

# bash math.sh

# Personalized --------------------------------
# set it 
BASE_DIR=/fs/ess/PGS0218/xli74/GOAT-PEFT #e.g. /home/xxx/GOAT-PEFT
OUT_DIR=/fs/ess/PGS0218/xli74/GOAT-PEFT/output #e.g. /mnt/models/
cd $BASE_DIR

set -xe

lora_dirs=()

# TOT_CUDA="0,1,2,3"
TOT_CUDA=$CUDA_VISIBLE_DEVICES
CUDAs=(${TOT_CUDA//,/ })
CUDA_NUM=${#CUDAs[@]}
run_command="CUDA_VISIBLE_DEVICES=$TOT_CUDA torchrun --standalone --nnodes=1 --nproc-per-node=$CUDA_NUM "
model='mistralai/Mistral-7B-Instruct-v0.2'
task=metamathqa100k
totalbz=32
rank=8
alpha=16
bz=4
gacc=$(( totalbz / bz / CUDA_NUM ))
ep=1
lr=2e-5
lora=rslora-pro
k=2
e=8
aux=1e-3

cd $BASE_DIR/goat
# conda activate goat    # 在slurm中已经加载了环境

MOE(){
echo "Using GPUs: $TOT_CUDA"
echo "Total GPU count: $CUDA_NUM"

cd $BASE_DIR/goat
export ETA=1.0
unset WANDB_MODE
if [ -n "$DEBUG" ]; then
  export WANDB_MODE=disabled
fi

lora=src.goat


prj=${model}-$task-${lora}${aux}-${k}in${e}-total${totalbz}dp${CUDA_NUM}bz${bz}lr${lr}
out="$OUT_DIR/$prj"

eval $run_command \
train_nlg_personalized1.py \
--model $model \
--lora $lora \
--aux_loss_coeff=$aux \
--experts=$e \
--k $k \
--task $task \
--bz $bz \
--gacc $gacc \
--ep $ep \
--lr $lr \
--prj $prj \
--rank $rank \
--alpha $alpha \
--output $out \
--seed 0 \
--result $BASE_DIR/goat/results/jianfeng_data_new1 \
--git_hash $(git rev-parse --short HEAD)

lora_dirs+=($prj)

}


MOE



