import os, sys
from os.path import dirname as up

sys.path.append(os.path.abspath(os.path.join(up(__file__), os.pardir)))

# Training Config
import torch
from contextlib import nullcontext
from torch.optim.lr_scheduler import LinearLR,SequentialLR, CosineAnnealingLR

import json

TRAINING_CONFIG_PATH = 'config/training_config.json'

with open(TRAINING_CONFIG_PATH, 'r') as f:
    training_config = json.load(f)

learning_rate = training_config["learning_rate"] #more stable training, earlier 1e-4
max_iters = training_config["max_iters"] #increase from 25000
warmup_steps = training_config["warmup_steps"] #smoother initial train, earlier 100
min_lr = training_config["min_lr"] #lower rate, earlier 5e-4
eval_iters = training_config["eval_iters"] # increased from 100
batch_size = training_config["batch_size"] # changed from 16, better gradient estimate
block_size = training_config["block_size"] #changed from 64, capture longer range dependencies

gradient_accumulation_steps = training_config["gradient_accumulation_steps"] # reduced from 50

device = training_config["device"] if torch.cuda.is_available() else "cpu"
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler

dtype = training_config["dtype"] if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)