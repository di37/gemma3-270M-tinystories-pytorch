import os, sys
from os.path import dirname as up

sys.path.append(os.path.abspath(os.path.join(up(__file__), os.pardir)))

import json

MODEL_CONFIG_PATH = 'config/model_config.json'

with open(MODEL_CONFIG_PATH, 'r') as f:
    model_config = json.load(f)

# print(model_config)



