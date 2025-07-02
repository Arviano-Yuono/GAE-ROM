# src/utils/commons.py
# common functions

import torch
import numpy as np
import random
import os
import logging
import yaml
import torch.nn.functional as F

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_activation(act_name: str):
    return getattr(torch.nn, act_name)()

def get_activation_function(act_name: str):
    return getattr(F, act_name.lower())


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


def save_config(config: dict, task: str):
    model_type = config['training']['model_name']
    num_epochs = config['training']['epochs']
    os.makedirs(f'artifacts/{task}/{model_type}', exist_ok=True)
    save_path = f"""artifacts/{task}/{model_type}/{model_type}_best_model_{num_epochs}_config.yaml"""
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


def is_scheduler_per_batch(scheduler):
    if scheduler is None:
        return False
    if (isinstance(scheduler, torch.optim.lr_scheduler.CyclicLR)
        or isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)
        or isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR)):
        return True
    else:
        return False


def load_model(save_path: str, config=None, ae_input_ize=None):
    # Load the complete state
    state = torch.load(save_path, weights_only=False)
    
    # Create a new model instance
    from src.model.gae import GAE
    
    # If config is not provided, use the saved config
    if config is None:
        config = state['save_config']
    
    model = GAE(config=config, ae_input_size=ae_input_ize)
    
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














