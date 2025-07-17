# data1, support goat
BASE_MODEL="mistralai/Mistral-7B-Instruct-v0.2"
OUTPUT="output"
LORA_RANK=64
LORA_METHOD="goat"  # lora / goat
DATA_DIR="../personalization_data/jianfeng_data_new1"  


python train_all_nonpersonalized_new.py \
    --model_name_or_path $BASE_MODEL \
    --fine_tuned_part_name_or_path $BASE_MODEL \
    --output_dir $OUTPUT \
    --lora_method $LORA_METHOD \
    --lora_r $LORA_RANK \
    --data_path $DATA_DIR/train.jsonl \
    --dataset_split "train" \
    --personalization_data_path $DATA_DIR/personalized_data.jsonl \
    --dataset_field instruction output \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 64 \
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
    --note "goat_rank64_alpha128_epoch2_b2" \


