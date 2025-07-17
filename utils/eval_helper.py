import re
from typing import Optional
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
from utils.data_helper import StopOnSubstrings




def extract_option_letter(output: str) -> Optional[str]:

    match = re.search(r"####\s*([A-Z])", output)
    return match.group(1) if match else None


