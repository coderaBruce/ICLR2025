#!/bin/bash
set -xe

# export CUDA_LAUNCH_BLOCKING=1  # 禁用异步CUDA, 好像能保证多卡训练可复现


# data1, support goat
BASE_MODEL="mistralai/Mistral-7B-Instruct-v0.2"
OUTPUT="output"
LORA_RANK=64
LORA_METHOD="goat"  # lora / goat
DATA_DIR="../personalization_data/jianfeng_data_new4" 

# === 分布式训练配置 ===
NNODES=${SLURM_JOB_NUM_NODES}
NODE_RANK=${SLURM_NODEID}
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
MASTER_PORT=64444

# gacc, 每节点 GPU 数量
ONE_CARD_GACC=64
CUDA_NUM=${SLURM_GPUS_ON_NODE}
run_command="torchrun --nnodes=$NNODES --nproc-per-node=$CUDA_NUM --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"
gacc=$(( ONE_CARD_GACC / (CUDA_NUM * NNODES) ))

# === 启动分布式训练 ===
eval $run_command \
  train_all_nonpersonalized_new_ddp.py \
    --model_name_or_path $BASE_MODEL \
    --fine_tuned_part_name_or_path $BASE_MODEL \
    --output_dir $OUTPUT \
    --lora_method $LORA_METHOD \
    --lora_r $LORA_RANK \
    --data_path $DATA_DIR/train.jsonl \
    --dataset_split "train" \
    --personalization_data_path $DATA_DIR/personalized_data.jsonl \
    --dataset_field instruction output \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps $gacc \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 5 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 True \
    --tf32 False \
    --fp16 False \
    --experts 8 \
    --aux_loss_coeff 1e-3 \
    --top_k 2 \
    --lora_alpha 128 \
    --note "goat_all_in_one_data4_epoch5" \


