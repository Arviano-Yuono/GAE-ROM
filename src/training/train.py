from src.model.gae import GAE
from src.training.val import val
from src.utils.commons import get_config, save_model
import torch
import torch_geometric
import torch.nn.functional as F

import tqdm


config = get_config('configs/default.yaml')['training']

def train(model: GAE, 
          device: torch.device, 
          train_loader: torch_geometric.loader.DataLoader,
          is_val: bool = False,
          val_loader: torch_geometric.loader.DataLoader = None,
          is_tqdm: bool = True,
          single_batch: bool = False,
          save_best_model: bool = True,
          config = config):
    
    torch.cuda.empty_cache()
    
    model_config = model.config
    model_name = f"""{model_config['encoder']['convolution_layers']['type']}_\
        {model_config['encoder']['pool']['type']}_\
            {model_config['encoder']['pool']['ratio']}_\
                {model_config['encoder']['pool']['is_pooling']}"""
    # loss
    loss_fn = config['loss']['type']
    if loss_fn == 'mse':
        loss_fn = F.mse_loss
    elif loss_fn == 'rmse':
        loss_fn = lambda x, y: torch.sqrt(F.mse_loss(x, y))
    elif loss_fn == 'l1_loss':
        loss_fn = F.l1_loss
    elif loss_fn == 'l2_loss':
        loss_fn = F.l2_loss
    elif loss_fn == 'smooth_l1_loss':
        loss_fn = F.smooth_l1_loss 
    else:
        raise ValueError(f"Invalid loss function: {loss_fn}")
    
    # optimizer
    if config['optimizer']['type'] == 'Adam':
        from torch.optim.adam import Adam
        optimizer = Adam(model.parameters(), lr=config['optimizer']['learning_rate'], weight_decay=config['optimizer']['weight_decay'])
    elif config['optimizer']['type'] == 'AdamW':
        from torch.optim.adamw import AdamW
        optimizer = AdamW(model.parameters(), lr=config['optimizer']['learning_rate'], weight_decay=config['optimizer']['weight_decay'])
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
    
    train_history = dict(train_loss=[], val_loss=[])
    best_loss = float('inf')
    loss_val = None
    # training loop
    model.train()

    if is_tqdm:
        loop = tqdm.tqdm(range(config['epochs']))
    else:
        loop = range(config['epochs'])

    for i in loop:
        # implement torch amp
        reconstruction_loss = torch.tensor(0., device=device)
        if config['amp']:
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                if single_batch:
                    batch = next(iter(train_loader))
                    batch = batch.to(device)
                    target = batch.x
                    optimizer.zero_grad()
                    out, _ = model(batch)
                    reconstruction_loss += F.mse_loss(input=out, target=target)
                    loss_train = reconstruction_loss
                    loss_train.backward()
                else:
                    for batch in train_loader:
                        batch = batch.to(device)
                        target = batch.x
                        optimizer.zero_grad()
                        out, _ = model(batch)
                        reconstruction_loss += F.mse_loss(input=out, target=target)
                    loss_train = reconstruction_loss / len(train_loader)  # Average the loss
                    loss_train.backward()
        else:
            if single_batch:
                batch = next(iter(train_loader))
                batch = batch.to(device)
                target = batch.x
                optimizer.zero_grad()
                out, _ = model(batch)
                reconstruction_loss += F.mse_loss(input=out, target=target)
                loss_train = reconstruction_loss
                loss_train.backward()
            else:
                for batch in train_loader:
                    batch = batch.to(device)
                    target = batch.x
                    optimizer.zero_grad()
                    out, _ = model(batch)
                    reconstruction_loss += F.mse_loss(input=out, target=target)
                loss_train = reconstruction_loss / len(train_loader)  # Average the loss
                loss_train.backward()
        # Add gradient clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if is_val:
            loss_val = val(model, device, val_loader)
            model.train()
            train_history['val_loss'].append(loss_val.item())

        # save best model
        if is_val:
            if loss_val.item() < best_loss and save_best_model:
                best_loss = loss_val.item()
                save_model(model, f'artifacts/{model_name}_best_model.pth')
        else:
            if loss_train.item() < best_loss and save_best_model:
                best_loss = loss_train.item()
                save_model(model, f'artifacts/{model_name}_best_model.pth')
        
        train_history['train_loss'].append(loss_train.item())

        if i != 0 and i % config['print_train'] == 0:
            if is_val:
                print(f"Epoch {i+1}/{config['epochs']}, train_loss: {loss_train.item()}, val_loss: {loss_val.item()}")
            else:
                print(f"Epoch {i+1}/{config['epochs']}, train_loss: {loss_train.item()}")
    
    return train_history
