# Load, tokenize, collate, etc.



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


IGNORE_INDEX = -100



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



def get_unpersonalized_training_dataset(
    args,
    tokenizer,
    batch_size: int = 3000,
    num_proc: int = 32,
    use_cache: bool = True,
) -> dict:
    """
    Use HF load_dataset to load the unpersonalized data; then use map to tokenize the data.

    参数:
        args: 包含 data_path, dataset_split, dataset_field 的参数对象
        tokenizer: 用于分词的 tokenizer 实例
        map_tokenize_function: tokenize 映射函数
        batch_size: 每个 batch 的大小（默认 3000)
        num_proc: 并行进程数（默认 32)
        use_cache: 是否使用缓存（默认 True)

    返回:
        预处理后的 Hugging Face Dataset 对象
    """
    raw_train_datasets = load_dataset("json", data_files=args.data_path, split=args.dataset_split)

    train_dataset = raw_train_datasets.map(
        function=map_tokenize_function,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=use_cache,
        desc="Running tokenizer on train dataset",
        fn_kwargs={
            "tokenizer": tokenizer,
            "query": args.dataset_field[0],
            "response": args.dataset_field[1],
        },
    )

    return train_dataset

