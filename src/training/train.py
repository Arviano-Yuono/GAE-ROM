from src.model.gae import GAE
from src.utils.commons import get_config, save_model, load_model
import torch
import torch_geometric
import torch.nn.functional as F
from torch_geometric.data import Data

import torch
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW

import tqdm

config = get_config('configs/default.yaml')['training']

def train(model, 
          device: torch.device, 
          train_loader: torch_geometric.loader.DataLoader,
          print_every = None,
          single_batch: bool = False,
          config = config):
    
    torch.cuda.empty_cache()
    
    # optimizer
    if config['optimizer']['type'] == 'Adam':
        from torch.optim.adam import Adam
        optimizer = Adam(model.parameters(), lr=config['optimizer']['learning_rate'])
    elif config['optimizer']['type'] == 'AdamW':
        from torch.optim.adamw import AdamW
        optimizer = AdamW(model.parameters(), lr=config['optimizer']['learning_rate'])
    else:
        raise ValueError(f"Invalid optimizer: {config['optimizer']['type']}")

    # scheduler 
    if config['scheduler']['type'] == 'StepLR':
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(optimizer, step_size=config['scheduler']['step_size'], gamma=config['scheduler']['gamma'])
    elif config['scheduler']['type'] == 'CosineAnnealingLR':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])
    else:
        raise ValueError(f"Invalid scheduler: {config['scheduler']['type']}")
    
    train_history = dict(train_loss=[])
    best_loss = float('inf')

    # training loop
    model.train()
    loop = tqdm.tqdm(range(config['epochs']))
    for i in loop:
        # implement torch amp
        if config['amp']:
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                reconstruction_loss = torch.tensor(0., device=device)
                if single_batch:
                    batch = next(iter(train_loader))
                    batch = batch.to(device)
                    target = batch.x
                    optimizer.zero_grad()
                    out, _ = model(batch)
                    reconstruction_loss += F.mse_loss(input=out, target=target)
                else:
                    for batch in train_loader:
                        batch = batch.to(device)
                        target = batch.x
                        optimizer.zero_grad()
                        out, _ = model(batch)
                        reconstruction_loss += F.mse_loss(input=out, target=target)
        else:
            reconstruction_loss = torch.tensor(0., device=device)
            if single_batch:
                batch = next(iter(train_loader))
                batch = batch.to(device)
                target = batch.x
                optimizer.zero_grad()
                out, _ = model(batch)
                reconstruction_loss += F.mse_loss(input=out, target=target)
            else:
                for batch in train_loader:
                    batch = batch.to(device)
                    target = batch.x
                    optimizer.zero_grad()
                    out, _ = model(batch)
                    reconstruction_loss += F.mse_loss(input=out, target=target)
        loss_train = reconstruction_loss / len(train_loader)  # Average the loss
        loss_train.backward()
        optimizer.step()
        scheduler.step()
        # save best model
        if loss_train.item() < best_loss:
            best_loss = loss_train.item()
            save_model(model, f'artifacts/{config["model_name"]}_best_model.pth')
        train_history['train_loss'].append(loss_train.item())
        if i != 0 and i % config['print_train'] == 0:
            print(f"Epoch {i+1}/{config['epochs']}, Loss: {loss_train.item()}")
    
    return train_history
