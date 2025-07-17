import os
import argparse
import logging
from pathlib import Path
import torch
from torch import Tensor, nn
from typing import List

from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
    TextGenerationPipeline,
    GenerationConfig,
    TrainerCallback,
)

from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.trainer_utils import get_last_checkpoint
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, set_seed,
    PreTrainedTokenizer, LlamaTokenizer, StoppingCriteria, StoppingCriteriaList
)
from datasets import Dataset, DatasetDict, load_dataset
import transformers
from transformers import TrainingArguments, Trainer
from transformers import default_data_collator
import importlib
import peft
from peft import LoraConfig, PeftModel, get_peft_model
from packaging import version
from transformers.training_args import TrainingArguments
from peta.utils import TitledLog
import torch.distributed as dist
import wandb
import tqdm 
import math
import re
import peta
log = logging.getLogger(__name__)

# disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_SILENT"] = "true"

torch.set_float32_matmul_precision("medium")

import pandas as pd
def evaluate_on_test_set(model, tokenizer, test_file, device="cuda"):

    from tqdm import tqdm
    import json

    # 加载测试集
    # dataset = load_dataset("json", data_files=test_file)
    dataset = load_dataset("json", data_files={"test": test_file})["test"]


    correct = 0
    total = 0

    for example in tqdm(dataset, desc="Evaluating"):
        prompt = example["instruction"]
        target = example["output"].strip()

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
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
                    # stopping_criteria=StoppingCriteriaList([
                    #     StopOnSubstrings(["Instruction:", "Response:", "Instruction", "Response"], tokenizer)
                    # ])
                )
        # decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


        # 获取生成的答案
        pred_answer = decoded.strip().split("####")[-1].strip().split()[0]
        gold_answer = target.strip().split("####")[-1].strip().split()[0]

        if pred_answer == gold_answer:
            correct += 1
        total += 1

    acc = correct / total
    print(f"\n🎯 Final Test Accuracy: {acc * 100:.2f}% ({correct}/{total})\n")


class StopOnSubstrings(StoppingCriteria):
    def __init__(self, stop_strs: List[str], tokenizer: PreTrainedTokenizer):
        self.stop_ids = [tokenizer.encode(s, add_special_tokens=False) for s in stop_strs]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_sequence in self.stop_ids:
            if input_ids.shape[1] >= len(stop_sequence):
                if input_ids[0, -len(stop_sequence):].tolist() == stop_sequence:
                    return True
        return False


def _is_peft_model(model):
    classes_to_check = (PeftModel,)
    # Here we also check if the model is an instance of `PeftMixedModel` introduced in peft>=0.7.0: https://github.com/huggingface/transformers/pull/28321
    if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
        from peft import PeftMixedModel
        classes_to_check = (*classes_to_check, PeftMixedModel)
    return isinstance(model, classes_to_check)



class CustomTrainer(Trainer):
    def __init__(self, *args, cus_args=None, scaling_factor=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.cus_args = cus_args
        if 'ft' not in cus_args.lora:
            self.scaling_factor = scaling_factor
        self.aux_loss_coeff = self.cus_args.aux_loss_coeff

    def compute_loss(self, model, inputs, return_outputs=False):

        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)

            if hasattr(unwrapped_model, 'get_aux_loss'):
                aux_loss = unwrapped_model.get_aux_loss()
                loss += self.aux_loss_coeff * aux_loss
            
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            unwrapped_model = self.accelerator.unwrap_model(model)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            if hasattr(unwrapped_model, 'get_aux_loss'):
                aux_loss = unwrapped_model.get_aux_loss()
                loss += self.aux_loss_coeff * aux_loss

        return (loss, outputs) if return_outputs else loss

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lora', default="lora-pro", type=str)
    parser.add_argument('--model', default="roberta-large", type=str)
    parser.add_argument('--result', default="", type=str)
    parser.add_argument('--prj', default="lora-pro", type=str)
    parser.add_argument('--task', default="math", type=str)
    parser.add_argument('--output', default="output", type=str)
    parser.add_argument('--experts', default=1, type=int)
    parser.add_argument('--aux_loss_coeff', default=1., type=float)
    parser.add_argument('--shared_experts', default=1, type=int)
    parser.add_argument('--k', default=1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', action='store_true', help="Resume training from the last checkpoint.")              
    parser.add_argument('--gradient_checkpointing', action='store_true', help="Resume training from the last checkpoint.")              
    parser.add_argument('--ep', default=1, type=int)
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--bz', default=1, type=int)
    parser.add_argument('--gacc', default=1, type=int)
    parser.add_argument('--warmup', default=0, type=int)
    parser.add_argument('--sched', default="cosine", type=str)
    parser.add_argument('--rank', default=8, type=int)
    parser.add_argument('--alpha', default=16, type=int)
    parser.add_argument('--git_hash', default='', type=str)
    parser.add_argument('--modules', type=str, default='qkvoudg', help='target modules in lora layers')
    args = parser.parse_args()
    # assert args.lora in ["lora-pro", "rslora-pro"]
    return args      

def main():
    
    args = get_arguments()
    
    log.info(f"set seed to {args.seed}")
    transformers.set_seed(args.seed)
    set_seed(args.seed)
    
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    
    if rank == 0:
        wandb.init(
            project=f'{args.model}-LoRA'.replace('/',''),  
            name=args.prj,
            config=args,
        )
    
    path =  args.model
    tokenizer = transformers.LlamaTokenizer.from_pretrained(path,) #  padding_side="left"
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "</s>"})
        model.resize_token_embeddings(len(tokenizer))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # flash_attn is deterministic under this version by setting env vars; view transformers/modeling_flash_attention_utils.py:L237
    os.environ['FLASH_ATTENTION_DETERMINISTIC'] = '1'
    model = transformers.LlamaForCausalLM.from_pretrained(
        path, 
        max_length=1024, 
        # attn_implementation="flash_attention_2",  # 据说影响推理速度
        torch_dtype=torch.bfloat16, 
        device_map={"": local_rank}
    )

    scaling_factor = 2
    if 'ft' != args.lora:
        lora_r = args.rank
        lora_alpha = args.alpha
        lora_module_maps = {
            "q": "q_proj",
            "k": "k_proj",
            "v": "v_proj",
            "o": "o_proj",
            "u": "up_proj",
            "d": "down_proj",
            "g": "gate_proj",
        }
        target_modules = [lora_module_maps[char] for char in args.modules]
        
        if 'src' in args.lora:

            import peta.src
            import peft.peft_model
            peft.peft_model.get_peft_model_state_dict = peta.src.utils.save_and_load.get_peft_model_state_dict
            peft.peft_model.set_peft_model_state_dict = peta.src.utils.save_and_load.set_peft_model_state_dict
            try:
                peft_type, init_type, init_cof = args.lora.split('.')
            except:
                peft_type, init_type = args.lora.split('.')
                init_cof = 1 / args.experts

            peft.peft_model.PEFT_TYPE_TO_MODEL_MAPPING.update(
                {"GOAT": peta.src.GOATModel }
            )
            lora_config = peta.src.GOATConfig(
                r=lora_r,
                use_rslora=True if "rs" in args.lora else False,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                target_modules=target_modules,
                task_type="CAUSAL_LM",
                bias="none",
                num_experts=args.experts,
                top_k=args.k,
                init_type=init_type,
                init_cof=float(init_cof),
            )
        if local_rank == 0:
            print('>>>', lora_config)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        if args.lora not in ["lora", "rslora"]: 
            for name, module in model.named_modules():
                if "lora_" in name:  
                    module.to(torch.float32)
        scaling_factor = (lora_config.lora_alpha / math.sqrt(lora_config.r)) if "rs" in args.lora else (lora_config.lora_alpha / lora_config.r)

    # Print Model Args, not necessary
    if local_rank == 0:
        unique_patterns = set()
        for n, p in model.named_parameters():
            if p.requires_grad:
                if '.layer.' in n:
                    names = n.split('.layer.')
                    n = names[0].replace('base_model.', '') + '.' + '.'.join(names[1].split('.')[1:])
                elif '.layers.' in n:
                    names = n.split('.layers.')
                    n = names[0].replace('base_model.', '') + '.' + '.'.join(names[1].split('.')[1:])
                unique_patterns.add(n)
        print(unique_patterns)
    
    with TitledLog("load datasets and dataloaders", log_fn=log.info):
        import peta
        # NOTE: you may need to rm -rf ./data_cache
       
        from datasets import load_dataset

        
        data_path = "../../personalization_data/jianfeng_data_new1/train.jsonl"
        raw_dataset = load_dataset("json", data_files={"train": data_path})


        def preprocess(example):
            full_prompt = example["instruction"]
            completion = example["output"]
            
            # tokenize whole prompt+answer
            tokenized = tokenizer(
                full_prompt,
                truncation=True,
                max_length=1024,
                padding=False,
            )
            labels = tokenizer(
                completion,
                truncation=True,
                max_length=128,
                padding=False,
            )["input_ids"]

            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]

            # Align labels at the end of the prompt
            label_ids = [-100] * len(input_ids) + labels
            input_ids = input_ids + labels
            attention_mask = attention_mask + [1] * len(labels)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": label_ids,
            }

        datasets = raw_dataset.map(
            preprocess,
            remove_columns=raw_dataset["train"].column_names,
        )


        
        if local_rank == 0:
            print('>>>', tokenizer.decode(datasets['train'][0]['input_ids']))
            print('>>>', tokenizer.decode([l for l in datasets['train'][0]['labels'] if l != -100]))
    
    
    trainer = CustomTrainer(
        scaling_factor=scaling_factor,  
        model=model,
        train_dataset=datasets["train"],
        eval_dataset=None,
        tokenizer=tokenizer,
        args=TrainingArguments(
            output_dir=args.output, 
            logging_dir="./transformers_logs",
            do_train=True,
            num_train_epochs=args.ep,
            per_device_train_batch_size=args.bz,
            gradient_accumulation_steps=args.gacc,
            optim="adamw_torch",
            logging_steps=1,
            bf16=True,
            learning_rate=args.lr,
            weight_decay=0, # No weight decay
            warmup_ratio=0.03, # warmup step override it 
            warmup_steps=args.warmup,
            lr_scheduler_type=args.sched,
            report_to="wandb" if rank == 0 else None, 
            label_names=["labels"],  
            ddp_find_unused_parameters=False if 'adamole' not in args.lora else True,
            gradient_checkpointing=args.gradient_checkpointing,
            per_device_eval_batch_size=1,
            evaluation_strategy="no",
            eval_steps=-1,
            save_strategy="no",
            save_steps=-1,
            save_total_limit=100,
            # deepspeed="./config/deepspeed_zero2.json" if world_size > 1  and 'ft' not in args.lora and 'adamole' not in args.lora else None, 
        ),
        # data_collator=default_data_collator,
        data_collator=DataCollatorForSeq2Seq(tokenizer, return_tensors="pt", padding=True),
        cus_args = args,
    )

    trainer.train(resume_from_checkpoint=args.resume)

    # 保存并合并模型后
    if rank==0:
        model.eval()
        evaluate_on_test_set(model, tokenizer, test_file="../../personalization_data/jianfeng_data_new1/test.jsonl")

    if rank ==  0:
        print(f'saved in {args.output}')
        wandb.finish()
            

if __name__ == "__main__":
    main()
