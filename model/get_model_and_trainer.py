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
from transformers import TrainerCallback
from datasets import load_dataset
from transformers import logging as hf_logging
# from peft_local.peft_0_14_0.src.peft import LoraConfig, PeftModel, get_peft_model
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DistributedSampler




class HybridPrecisionTrainer(Trainer):
    """
    混合精度训练器：
    - 正常情况下使用FP16 (快速)
    - 检测到梯度异常时自动切换到BF16 (稳定)
    - 稳定几步后再切回FP16
    """
    
    def __init__(self, *args,  cus_args=None, scaling_factor=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.precision_mode = "fp16"  # 默认使用FP16
        self.unstable_count = 0
        self.stable_count = 0
        self.switch_threshold = 3  # 连续3次异常就切换
        self.recovery_threshold = 10  # 连续10次稳定就切回
        self.max_grad_norm_threshold = 10.0  # 梯度正常的阈值

        self.cus_args = cus_args
        if 'ft' not in cus_args.lora_method:
            self.scaling_factor = scaling_factor
        self.aux_loss_coeff = self.cus_args.aux_loss_coeff
        
        logger = logging.getLogger(__name__)
        logger.info(f"HybridPrecisionTrainer initialized")
        logger.info(f"Device supports BF16: {torch.cuda.is_bf16_supported()}")
    
    def training_step(self, model, inputs):
        """重写训练步骤，实现动态精度切换"""
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        # 根据当前精度模式选择autocast
        if self.precision_mode == "bf16":
            autocast_dtype = torch.bfloat16
        else:
            autocast_dtype = torch.float16
        
        with torch.amp.autocast(dtype=autocast_dtype, enabled=True):
            loss = self.compute_loss(model, inputs)
        
        if self.args.n_gpu > 1:
            loss = loss.mean()
        
        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            loss = loss / self.args.gradient_accumulation_steps
        
        # 根据精度模式选择scaler
        if self.precision_mode == "bf16":
            # BF16通常不需要loss scaling
            loss.backward()
        else:
            # FP16需要loss scaling
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
        
        return loss.detach()
    
    def _check_gradient_health(self, model):
        """检查梯度健康状况"""
        total_norm = 0.0
        param_count = 0
        nan_count = 0
        inf_count = 0
        
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                if torch.isnan(param_norm):
                    nan_count += 1
                elif torch.isinf(param_norm):
                    inf_count += 1
                else:
                    total_norm += param_norm.item() ** 2
                param_count += 1
        
        if param_count > 0:
            total_norm = total_norm ** (1. / 2)
        
        return {
            'grad_norm': total_norm,
            'nan_count': nan_count,
            'inf_count': inf_count,
            'param_count': param_count,
            'is_healthy': nan_count == 0 and inf_count == 0 and total_norm < self.max_grad_norm_threshold
        }
    
    def _maybe_switch_precision(self, grad_health):
        """根据梯度健康状况决定是否切换精度"""
        old_mode = self.precision_mode
        
        if not grad_health['is_healthy']:
            self.unstable_count += 1
            self.stable_count = 0
            
            # 如果当前是FP16且连续不稳定，切换到BF16
            if self.precision_mode == "fp16" and self.unstable_count >= self.switch_threshold:
                self.precision_mode = "bf16"
                self.unstable_count = 0
                logger = logging.getLogger(__name__)
                logger.warning(f"Switched to BF16 due to gradient instability")
                
                # 重新初始化scaler for BF16
                if hasattr(self, 'scaler'):
                    self.scaler = None
        else:
            self.stable_count += 1
            self.unstable_count = 0
            
            # 如果当前是BF16且连续稳定，切换回FP16
            if self.precision_mode == "bf16" and self.stable_count >= self.recovery_threshold:
                self.precision_mode = "fp16"
                self.stable_count = 0
                logger = logging.getLogger(__name__)
                logger.info(f"Switched back to FP16 - gradients are stable")
                
                # 重新初始化scaler for FP16
                if not hasattr(self, 'scaler') or self.scaler is None:
                    self.scaler = torch.amp.GradScaler()
        
        if old_mode != self.precision_mode:
            return True
        return False
    
    def training_step(self, model, inputs):
        """重写训练步骤"""
        loss = super().training_step(model, inputs)

        # 梯度裁剪
        if self.precision_mode != "bf16":
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 检查梯度健康状况
        grad_health = self._check_gradient_health(model)
        
        # 决定是否切换精度
        switched = self._maybe_switch_precision(grad_health)
        
        # 记录信息
        if self.state.global_step % self.args.logging_steps == 0:
            logger = logging.getLogger(__name__)
            logger.info(f"Step {self.state.global_step}: "
                       f"Mode={self.precision_mode}, "
                       f"GradNorm={grad_health['grad_norm']:.4f}, "
                       f"NaN={grad_health['nan_count']}, "
                       f"Inf={grad_health['inf_count']}")
        
        return loss
    
    def compute_loss(self, model, inputs, return_outputs=False):
        
        outputs = model(**inputs)

        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        if hasattr(self, "accelerator") and self.accelerator is not None: # Incase accelerator is not installed
            unwrapped_model = self.accelerator.unwrap_model(model)
        else:
            unwrapped_model = model  # fallback to original model
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        if hasattr(unwrapped_model, 'get_aux_loss'):
            aux_loss = unwrapped_model.get_aux_loss()
            loss += self.aux_loss_coeff * aux_loss

        return (loss, outputs) if return_outputs else loss


class DetectAnomalyCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        torch.autograd.set_detect_anomaly(True)

class GOATTrainer(Trainer):
    def __init__(self, *args, cus_args=None, scaling_factor=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.cus_args = cus_args
        if 'ft' not in cus_args.lora_method:
            self.scaling_factor = scaling_factor
        self.aux_loss_coeff = self.cus_args.aux_loss_coeff

    def compute_loss(self, model, inputs, return_outputs=False):
        
        outputs = model(**inputs)

        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        if hasattr(self, "accelerator") and self.accelerator is not None: # Incase accelerator is not installed
            unwrapped_model = self.accelerator.unwrap_model(model)
        else:
            unwrapped_model = model  # fallback to original model
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        if hasattr(unwrapped_model, 'get_aux_loss'):
            aux_loss = unwrapped_model.get_aux_loss()
            loss += self.aux_loss_coeff * aux_loss

        return (loss, outputs) if return_outputs else loss












def get_adapted_model(args, 
                    base_model,
                    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
                    ):
    # Vanilla LoRA
    if args.lora_method == "lora":
        lora_config = LoraConfig(
                        r=args.lora_r,
                        lora_alpha=args.lora_r,
                        target_modules=target_modules,
                        lora_dropout=0,
                        task_type="CAUSAL_LM",
                        use_dora=False,
                    )
        return get_peft_model(base_model, lora_config)

        

    if args.lora_method == "goat":
        
        lora_r = args.lora_r
        lora_alpha = args.lora_alpha


        import model.peta.src
        import peft.peft_model
        peft.peft_model.get_peft_model_state_dict = model.peta.src.utils.save_and_load.get_peft_model_state_dict
        peft.peft_model.set_peft_model_state_dict = model.peta.src.utils.save_and_load.set_peft_model_state_dict
        try:
            peft_type, init_type, init_cof = args.lora_method.split('.')
        except:
            # peft_type, init_type = args.lora_method.split('.')
            init_type = args.lora_method
            init_cof = 1 / args.experts

        peft.peft_model.PEFT_TYPE_TO_MODEL_MAPPING.update(
                {"GOAT": model.peta.src.GOATModel }
            )   
        lora_config = model.peta.src.GOATConfig(
            r=lora_r,
            use_rslora=True if "rs" in args.lora_method else False,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=target_modules,
            task_type="CAUSAL_LM",
            bias="none",
            num_experts=args.experts,
            top_k=args.top_k,
            init_type=init_type,
            init_cof=float(init_cof),
        )
        return get_peft_model(base_model, lora_config)

        




def get_adapted_trainer(
    args,
    model,
    tokenizer,
    train_dataset,
    data_collator,
):
    """
    构建并返回一个适配后的 Hugging Face Trainer 实例。

    参数:
        model: 要训练的模型（通常带有 LoRA/PEFT adapter）
        tokenizer: 用于训练的 tokenizer
        args: transformers.TrainingArguments 实例
        train_dataset: 训练数据集
        data_collator: 数据整理函数（如 SupervisedDataCollator）

    返回:
        transformers.Trainer 实例
    """
    if args.lora_method == "lora":
        return Trainer(
            model=model,
            tokenizer=tokenizer,
            args=args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
    

    if args.lora_method == "goat":
        scaling_factor = args.lora_alpha / args.lora_r
        return GOATTrainer(
            scaling_factor=scaling_factor,
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=None,
            data_collator=data_collator,
            args=args,
            cus_args = args,
            # callbacks=[DetectAnomalyCallback()],
        )
    
        # return HybridPrecisionTrainer(
        #         scaling_factor=scaling_factor,
        #         model=model,
        #         tokenizer=tokenizer,
        #         train_dataset=train_dataset,
        #         eval_dataset=None,
        #         data_collator=data_collator,
        #         args=args,
        #         cus_args = args,
        #         # callbacks=[DetectAnomalyCallback()],
        #     )
