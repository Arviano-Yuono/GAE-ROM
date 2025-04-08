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


def save_model(model: torch.nn.Module, save_path: str):
    # Save the model's state dictionary and additional properties
    state = {
        'state_dict': model.state_dict(),
        'save_config': model.save_config,
        'config': model.config,
    }
    torch.save(state, save_path)


def load_model(save_path: str):
    # Load the complete state
    state = torch.load(save_path, weights_only=False)
    
    # Create a new model instance
    from src.model.gae import GAE
    model = GAE(config = state['save_config'])
    
    # Load the state dictionary with strict=False to ignore mismatched keys
    model.load_state_dict(state['state_dict'], strict=False)
    
    # Load additional properties
    model.save_config = state['save_config']
    model.config = state['config']
    
    return model


def reset_imports():
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import src.model.gae
    import src.data.loader
    import src.utils.commons














