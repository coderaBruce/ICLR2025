# From train_all_nonpersonalized.py
# TO build a unified train script for MoE structure

# New: 
# 想要refactor一下
#TODO:
# 1. 好像没有必要先把data 用 Load_dataset 成hugginface dataset形式,再map, 直接参考aplaud做法

import os
import re
import json
import argparse
import datetime
import logging
import time
import math
from dataclasses import dataclass, field
from typing import List, Dict, Sequence, Optional
import pytz
import sys
import logging
import time
import os
import datetime
import traceback
import hashlib



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset
import transformers
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, set_seed,
    PreTrainedTokenizer, LlamaTokenizer, StoppingCriteria, StoppingCriteriaList
)
from datasets import load_dataset
from transformers import logging as hf_logging
# from peft_local.peft_0_14_0.src.peft import LoraConfig, PeftModel, get_peft_model
from peft import LoraConfig, PeftModel, get_peft_model


# From This Project
from utils.logger_helper import *
from utils.eval_helper import *
from utils.data_helper import *
from utils.train_helper import prepare_model_save_dir
from model.get_model_and_trainer import get_adapted_model, get_adapted_trainer

# APLAUD_LAYER_CLS = Aplaud.APlaudLinear_Cu_low_rank_post_with_resid_dora_half_ablation




@dataclass
class CustomArgs(TrainingArguments):
    # Unpersonalized args
    model_name_or_path: Optional[str] = field(default="mistralai/Mistral-7B-Instruct-v0.2", 
                                             metadata={"help": "Name/Path to Base Model (at least merged), must have tokenizer_config.json and tokenizer.model."}
                                             )
    fine_tuned_part_name_or_path: Optional[str] = field(default="mistralai/Mistral-7B-Instruct-v0.2",
                                                        metadata={"help": "path to the unmerged PEFT result, usually ended with ft"}
                                                        )
    adapter_name_or_path: Optional[str] = field(default=None)
    data_path: str = field(default=None, metadata={"help": "Path to the non personalized training data."})
    dataset_split: str = field(default="train[:100000]", metadata={"help": "dataset split"})
    dataset_field: List[str] = field(default=None, metadata={"help": "Fields of dataset input and output."})
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=4096, metadata={"help": "Maximum input sequence length."})
    lora_r: int = field(default=64)
    lora_method: str = field(default="lora", metadata={"help": "lora | lora-xs | goat"})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Maximum gradient norm for gradient clipping."})
    

    # personalized args
    output_dir: str = field(default="./outputs")
    personalization_data_path: str = field(default=None, metadata={"help": "Path to the personalization data."})
    per_device_train_batch_size: int = field(default=8)
    num_train_epochs: int = field(default=2)
    learning_rate: float = field(default=1e-4)
    bf16: bool = field(default=False)
    logging_steps: int = field(default=20)
    save_strategy: str = field(default="no")
    report_to: str = field(default="none")
    remove_unused_columns: bool = field(default=False)
    seed: int = field(default=42)
    applaud_mode: str = field(default="full")
    shared_dir: str = field(default="./shared")

    # goat args
    experts: int = field(default=1, metadata={"help": "Number of experts."})
    aux_loss_coeff: float = field(default=1., metadata={"help": "Auxiliary loss coefficient."})
    shared_experts: int = field(default=1, metadata={"help": "Number of shared experts."})
    top_k: int = field(default=1, metadata={"help": "Number of top_k experts."})
    lora_alpha: int = field(default=16, metadata={"help": "Lora alpha."})
    # Note
    note: str = field(default="")



def main(args, logger):

    logger.info("Start training...")
    set_seed(args.seed)

    # 1. Tokenizer + Train Data
    # 1-1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, 
        model_max_length=args.model_max_length,
        padding_side="right", 
        trust_remote_code=True,
        use_fast=True
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # 1-2. Train Data
    train_dataset = get_unpersonalized_training_dataset(
        args=args,  # args.data_path, args.dataset_split, args.dataset_field
        tokenizer=tokenizer,
        batch_size=3000,
        num_proc=32,
        use_cache=True,
    )

    # 2. Base Model + PEFT Model
    # 2-1. Base Model
    # must set use_cache = False for training
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    base_model.config.use_cache = False
    base_model.config.pad_token_id = tokenizer.pad_token_id
    
    # 2-2. PEFT Model
    model = get_adapted_model(
        args, 
        base_model, 
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # model = PeftModel.from_pretrained(base_model, args.fine_tuned_part_name_or_path)

    # set output_dir_this_run: fined tuned model output dir
    args.output_dir_this_run = prepare_model_save_dir(args)
    os.makedirs(args.output_dir_this_run, exist_ok=True)


    # DEBUG: print model structure
    # for name, module in model.named_modules():
    #     print('name', name)
    #     print('module', module)


    # 3. Trainer(collator) and Train
    # Double check use_cache=False. NECESSARY if we do monkey patching to model
    model.config.use_cache = False
    # trainer = Trainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     args=args,
    #     train_dataset=train_dataset,
    #     data_collator=SupervisedDataCollator(tokenizer)
    # )
    trainer = get_adapted_trainer(
        args=args,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        data_collator=SupervisedDataCollator(tokenizer),
    )


    # 强制使用cuDNN for many operations
    torch.backends.cudnn.enabled = True # 好像这个会稍微快一丢丢: 7.5h -> 7.0h for epoch2 on data1
    
    trainer.train()
    trainer.save_state()

    # If mergeable, merge and save
    # model.save_pretrained(os.path.join(args.output_dir_this_run, 'ft'))
    # model = model.merge_and_unload()
    # model.save_pretrained(os.path.join(args.output_dir_this_run, 'ft_merged'))
        
    
    # 4. Eval
    # model = PeftModel.from_pretrained(base_model, "/fs/ess/PGS0218/xli74/LoRA-XS/output/mistralai/Mistral-7B-Instruct-v0.2/jianfeng_data_new1/train.jsonl_split_train/lora_rank_64_lr_0.0002_seed_42/output_2025-05-04T11:01:57-24/ft")
    # Eval Per User Start------------------------------------------------------------
    with open(args.personalization_data_path) as f:
        user_data_list = [json.loads(line) for line in f]
    all_preds = []
    for user in tqdm(user_data_list[:200], desc="Per-user training and evaluation on personalization data", leave=False):       
        
        logger.info("Eval for user: %s", user['user_id'])
        model.eval()
        test_prompts = [q["input"] for q in user["test"]]
        test_golds = [q["gold"] for q in user["test"]]
        test_ids = [q["id"] for q in user["test"]]

        preds = []
        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad(), torch.autocast("cuda"):
                outputs = model.generate(
                    **inputs,   # automatically pass the input_ids, attention_mask since tokenizer returns dict. Just that attention_mask is all 1s
                    do_sample=False,
                    temperature=0.0,
                    top_k=10,
                    top_p=1,
                    max_new_tokens=4196,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    stopping_criteria=StoppingCriteriaList([
                        StopOnSubstrings(["Instruction:", "Response:", "Instruction", "Response"], tokenizer)
                    ])
                )
            # decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # pred = decoded[len(prompt):].strip()

            # 据说这么做robust, 但是没试过
            prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
            generated_ids = outputs[0][len(prompt_ids):]
            pred = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


            
            preds.append(pred)
            print('\npred----------------------------------------', pred)

        for qid, pred, gold in zip(test_ids, preds, test_golds):
            all_preds.append({"id": qid, "output": pred, "gold": gold})

    y_pred = [extract_option_letter(x["output"]) for x in all_preds]
    y_true = [extract_option_letter(x["gold"]) for x in all_preds]

    logger.info(f"y_pred: {y_pred}")
    logger.info(f"y_true: {y_true}")

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    logger.info(f"\n✅ Final Evaluation:\nAccuracy: {acc:.4f} | Macro-F1: {f1:.4f}")

    # Eval Per User End------------------------------------------------------------


if __name__ == "__main__":
    # Step 1: Parse args early
    parser = transformers.HfArgumentParser(CustomArgs)
    args, = parser.parse_args_into_dataclasses()
    # Step 2: Setup logger early
    logger = setup_logger(args)

    # Step 3: Redirect stdout/stderr to file
    log_path = logger.handlers[0].baseFilename
    # if get_rank() == 0:
    #     tee = Tee(log_path)
    # else:
    #     tee = None  # 不重定向 stdout

    
    logger.info("\n====== ARGUMENTS ======")
    for arg_name, arg_value in vars(args).items():
        logger.info(f"{arg_name:<20} : {arg_value}")
    logger.info("========================\n")


    tee = Tee(log_path)

    try:
        main(args, logger)
    except Exception as e:
        print("⚠️ Exception occurred:", repr(e))
        traceback.print_exc()  # <-- 这行非常关键，打印 traceback 到 stdout（也进日志）
        raise  # 可选：再次抛出，终止程序
    finally:
        tee.close()