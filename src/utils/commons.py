# src/utils/commons.py
# common functions

import torch
import numpy as np
import random
import os
import logging
import yaml



def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger


def get_config(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: dict, save_path: str):
    with open(save_path, 'w') as f:
        yaml.dump(config, f)








