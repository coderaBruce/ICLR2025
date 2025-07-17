# From Aplaud_main_test2_2_ablation.py
# TO build a unified train script for MoE structure

#TODO:
# 1. 好像没有必要先把data 用Load_dataset 成hugginface dataset形式,再map, 直接参考aplaud做法




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
from peft_local.peft_0_14_0.src.peft import LoraConfig, PeftModel, get_peft_model
# from peft import LoraConfig, PeftModel, get_peft_model

from utils.logger_helper import *
from utils.eval_helper import *

BASE_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
IGNORE_INDEX = -100
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
    lora_r: int = field(default=None)
    lora_method: str = field(default="lora", metadata={"help": "lora | lora-xs"})
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


    # Note
    note: str = field(default="")




def tokenize_pair(sources, targets, tokenizer):
    """
    是Aplaud personzliation data的方法

    Tokenize batches pair/single pair):
    Batches of single source target text pair for training
    一个batch就是一个list, 里面有多条数据, 每条数据是一个source和target的pair

    Args:
        sources (List[str]): List of source text strings
        targets (List[str]): List of target text strings 
        tokenizer: The tokenizer to use for encoding text

    Returns:
        dict: Dictionary containing:
            - input_ids: List of tokenized input sequences
            - labels: List of label sequences where source tokens are masked with IGNORE_INDEX

    Example:
        >>> sources = ["Question: What is 2+2?", "Question: What is 3+3?"]
        >>> targets = ["Answer: 4", "Answer: 6"]
        >>> result = tokenize_pair(sources, targets, tokenizer)
        >>> print(result["input_ids"][0])  # Combined tokenized sequence
        >>> print(result["labels"][0])     # Labels with source tokens masked
    """
    examples = [s + t for s, t in zip(sources, targets)]
    input_ids, labels = [], []
    for src, ex in zip(sources, examples):
        tok_ex = tokenizer(ex, max_length=tokenizer.model_max_length, truncation=True, padding=False, return_tensors="pt")
        tok_src = tokenizer(src, max_length=tokenizer.model_max_length, truncation=True, padding=False, return_tensors="pt")
        ids = tok_ex["input_ids"][0]
        label = ids.clone()
        label[:tok_src["input_ids"].shape[1]] = IGNORE_INDEX
        input_ids.append(ids)
        labels.append(label)
    return {"input_ids": input_ids, "labels": labels}

def map_tokenize_function(examples, tokenizer, query, response):
    """
        HF Dataset 的map 的输入输出和tokenize_pair 的输入输出都不一致
        HF Dataset.map 接受的map函数要求 
            输入是个dict, 但是tokenize_pair 的输入是list, 
            输出每个dict元素是List[List], 而tokenize_pair 的输出是list[Tensor]所以需要转换一下
        query, response 是train data jsonl的field name, 对应 instruction 和 output
    """
    
    sources = examples[query]
    targets = [t + tokenizer.eos_token for t in examples[response]]  # !!!Extremly important since it allows the model to learn when to stop
    result = tokenize_pair(sources, targets, tokenizer)

    # 转换 tensor → list
    result["input_ids"] = [x.tolist() if isinstance(x, torch.Tensor) else x for x in result["input_ids"]]
    result["labels"] = [x.tolist() if isinstance(x, torch.Tensor) else x for x in result["labels"]]

    return result




class SupervisedDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ex["input_ids"]) for ex in instances],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ex["labels"]) for ex in instances],
            batch_first=True,
            padding_value=IGNORE_INDEX
        )
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id)
        }

class StopOnSubstrings(StoppingCriteria):
    def __init__(self, stop_strs: List[str], tokenizer: PreTrainedTokenizer):
        self.stop_ids = [tokenizer.encode(s, add_special_tokens=False) for s in stop_strs]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_sequence in self.stop_ids:
            if input_ids.shape[1] >= len(stop_sequence):
                if input_ids[0, -len(stop_sequence):].tolist() == stop_sequence:
                    return True
        return False






def main(args, logger):


    logger.info("Start training...")
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME, 
        model_max_length=args.model_max_length,
        padding_side="right", 
        trust_remote_code=True,
        use_fast=True
    )
    # tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL_NAME, padding_side="right", trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # base_model outside the loop, must set use_cache = False for training
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    base_model.config.use_cache = False
    base_model.config.pad_token_id = tokenizer.pad_token_id
    

    if args.lora_r is not None:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0,
            task_type="CAUSAL_LM",
            use_dora=False,
        )
        model = get_peft_model(base_model, lora_config)
    else:
        raise ValueError("LoRA rank should be provided.")
    # model = PeftModel.from_pretrained(base_model, args.model_name_or_path)

    now = datetime.datetime.now()
    timestamp = now.strftime('%Y-%m-%d')  # 更简洁

    note_suffix = args.note.replace("/", "-") if args.note else "nonote"
    output_base_dir = os.path.join(
        args.output_dir,
        args.model_name_or_path,
        os.path.basename(os.path.dirname(args.data_path)),
        "ablation",
        f"{timestamp}_{note_suffix}"
    )
    # set output_dir and logging_dir
    args.output_dir = output_base_dir
    args.logging_dir = os.path.join(output_base_dir, "runs")
    os.makedirs(args.output_dir, exist_ok=True)


    # DEBUG: print model structure
    # for name, module in model.named_modules():
    #     print('name', name)
    #     print('module', module)



    raw_train_datasets = load_dataset("json", data_files=args.data_path, split=args.dataset_split)
    train_dataset = raw_train_datasets.map(
        map_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset",
        fn_kwargs={   # This function defines the arguments for map_tokenize_function
            "tokenizer": tokenizer,
            "query": args.dataset_field[0],
            "response": args.dataset_field[1],
        },
    )

    # Double check. NECESSARY if we do monkey patching to model
    model.config.use_cache = False
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=train_dataset,
        data_collator=SupervisedDataCollator(tokenizer)
    )
    

    trainer.train()
    trainer.save_state()
    model.save_pretrained(os.path.join(args.output_dir, 'ft'))
    
    
    # model = PeftModel.from_pretrained(base_model, "/fs/ess/PGS0218/xli74/LoRA-XS/output/mistralai/Mistral-7B-Instruct-v0.2/jianfeng_data_new1/train.jsonl_split_train/lora_rank_64_lr_0.0002_seed_42/output_2025-05-04T11:01:57-24/ft")
    model = model.merge_and_unload()
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
                    **inputs,
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



    tee = Tee(log_path)

    try:
        main(args, logger)
    except Exception as e:
        print("⚠️ Exception occurred:", repr(e))
        traceback.print_exc()  # <-- 这行非常关键，打印 traceback 到 stdout（也进日志）
        raise  # 可选：再次抛出，终止程序
    finally:
        tee.close()