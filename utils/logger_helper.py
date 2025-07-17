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
from pathlib import Path

import traceback
import hashlib
from transformers import logging as hf_logging





class Tee:
    def __init__(self, file_path):
        self.file = open(file_path, "a")   # append mode， avoid overwrite conflict with Tee
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self


    def write(self, data):
        if data.strip() != "":
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            line = f"{timestamp} - PRINT - {data}"
        else:
            line = data
        self.file.write(line)
        self.file.flush()
        self.stdout.write(line)  # both consoel and file have the time stamp
        self.stdout.flush()

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.file.close()

def est_time_converter(*args):
    tz = pytz.timezone("America/New_York")
    return datetime.datetime.now(tz).timetuple()


def setup_logger(args, log_dir: str = "./saved_logs") -> logging.Logger:
    model_name = args.model_name_or_path
    data_name = args.data_path

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    formatter.converter = est_time_converter  # EST

    tz = pytz.timezone("America/New_York")
    date_str = datetime.datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")

    model_tag = model_name.replace("/", "_").replace(".", "_")
    # data_tag = data_name.replace("/", "_").replace(".", "_")
    data_tag = Path(data_name).parents[0].name.replace(".", "_")  # Only use secondlast layer of the path, e.g., data1/train.jsonl -> data1

    full_log_dir = os.path.join(log_dir, data_tag, model_tag)
    os.makedirs(full_log_dir, exist_ok=True)
    # log_path = os.path.join(full_log_dir, f"{date_str}.log")
    note_suffix = f"_{args.note.strip().replace(' ', '_')}" if args.note else "no_note"
    log_path = os.path.join(full_log_dir, f"{date_str}{note_suffix}.log")


    logger = logging.getLogger("training_logger")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(log_path, mode="a")  # append mode， avoid overwrite conflict with Tee
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    logger.info(f"Logging to file: {log_path}")
    return logger


def redirect_transformers_logging_to(logger: logging.Logger):
    hf_logger = hf_logging.get_logger()
    hf_logger.handlers = []
    for handler in logger.handlers:
        hf_logger.addHandler(handler)
    hf_logger.setLevel(logger.level)