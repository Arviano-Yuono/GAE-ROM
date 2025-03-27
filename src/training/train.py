from src.model.gae import GAE
from src.utils import commons
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW

import tqdm

config = commons.get_config('configs/default.yaml')

def train(model, optimizer: torch.optim.Optimizer, 
          device: torch.device, 
          scheduler: torch.optim.lr_scheduler._LRScheduler, 
          train_loader: torch_geometric.loader.DataLoader,
          config = config):
    
    # evaluation variables
    train_history = dict(train_loss=[])

    # training loop
    model.train()
    loop = tqdm.tqdm(range(config['training']['epochs']))
    for i in loop:
        reconstruction_loss = torch.tensor(0., device=device)
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            reconstruction_loss += F.mse_loss(input=out, target=batch.y)
        loss_train = reconstruction_loss / len(train_loader)  # Average the loss
        loss_train.backward()
        optimizer.step()
        scheduler.step()
        train_history['train_loss'].append(loss_train.item())
        if i % config['training']['print_train'] == 0:
            print(f"Epoch {i+1}/{config['training']['epochs']}, Loss: {loss_train.item()}")
    
    return train_history
