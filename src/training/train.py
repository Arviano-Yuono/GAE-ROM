import os
import pickle
from src.model.gae import GAE
from src.training.val import val
from src.utils.commons import get_config, save_model, is_scheduler_per_batch
import torch
import torch_geometric
import torch.nn.functional as F
import torch.nn as nn

import tqdm


config = get_config('configs/default.yaml')['training']

def train(model: GAE, 
          device: torch.device, 
          train_loader: torch_geometric.loader.DataLoader,
          is_val: bool = False,
          val_loader: torch_geometric.loader.DataLoader = None,
          is_boudary = False,
          is_tqdm: bool = True,
          minibatch: bool = True,
          save_best_model: bool = True,
          save_history: bool = True,
          start_up_epoch: int = 50,
          config = config):
    
    torch.cuda.empty_cache()
    # train_trajectories = [x - 1 for x in train_loader.dataset.file_index]
    # train_params = params[train_trajectories].to(device)

    model_config = model.config
    model_name = f"""{model_config['encoder']['convolution_layers']['type']}"""
    # loss
    loss_fn = config['loss']['type']
    if loss_fn == 'mse':
        loss_fn = nn.MSELoss(reduction= 'none')
    else:
        try:
            loss_fn = getattr(nn, loss_fn)
        except AttributeError:
            raise ValueError(f"Invalid loss function: {loss_fn}")
    # optimizer
    try:
        optimizer_class = getattr(torch.optim, config['optimizer']['type'])
        optimizer = optimizer_class(
            model.parameters(), 
            lr=config['optimizer']['learning_rate'], 
            weight_decay=config['optimizer']['weight_decay']
        )
    except AttributeError:
        raise ValueError(f"Invalid optimizer: {config['optimizer']['type']}")

    # scheduler
    try:
        scheduler_class = getattr(torch.optim.lr_scheduler, config['scheduler']['type'])
        if config['scheduler']['type'] == 'StepLR':
            scheduler = scheduler_class(optimizer, step_size=config['scheduler']['step_size'], gamma=config['scheduler']['gamma'])
        elif config['scheduler']['type'] == 'CosineAnnealingLR':
            scheduler = scheduler_class(optimizer, T_max=config['epochs'])
        elif config['scheduler']['type'] == 'MultiStepLR':
            scheduler = scheduler_class(optimizer, milestones=config['scheduler']['milestones'], gamma=config['scheduler']['gamma'])
    except AttributeError:
        raise ValueError(f"Invalid scheduler: {config['scheduler']['type']}")
    
    train_history = dict(train_loss=[], map_loss=[], reconstruction_loss=[], surface_var_loss=[])
    val_history = dict(val_loss=[], map_loss=[], reconstruction_loss=[], surface_var_loss=[])
    best_loss = float('inf')
    loss_val = None


    # training loop
    if is_tqdm:
        loop = tqdm.tqdm(range(config['epochs']))
    else:
        loop = range(config['epochs'])

    for i in loop:
        model.train()
        # implement torch amp
        surface_var_loss = torch.tensor(0., device=device) 
        reconstruction_loss = torch.tensor(0., device=device)
        map_loss = torch.tensor(0., device=device)
        total_loss = torch.tensor(0., device=device)
        total_loss_train = 0
        surface_var_loss_cumulative = 0
        reconstruction_loss_cumulative = 0
        map_loss_cumulative = 0
        total_loss_cumulative = 0
        total_batches = 0
        start_ind = 0

        for batch in train_loader:
            optimizer.zero_grad()
            # Move batch to device and ensure correct data type
            batch = batch.to(device)
            target = batch.x.float()
            batch.x = batch.x.float()
            
            if config['amp']:
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                    out, latent_var, est_latent_var = model(batch)
            else:
                out, latent_var, est_latent_var = model(batch)
            
            start_ind += batch.batch_size

            # Calculate losses
            surface= batch.surf.bool().to(device)

            if config['lambda_surface'] > 0:
                surface_var_loss = F.mse_loss(input=out[surface, :], target=target[surface, :])
                volume_var_loss = F.mse_loss(input=out[~surface, :], target=target[~surface, :])
                reconstruction_loss = surface_var_loss * config['lambda_surface'] + volume_var_loss
            else:
                reconstruction_loss = F.mse_loss(input=out, target=target)
            

            map_loss = F.mse_loss(est_latent_var, latent_var)
            # surface_var_loss = loss_fn(input=out[surface, :], target=target[surface, :]).mean(dim = 0).mean()
            # reconstruction_loss = loss_fn(input=out[~surface, :], target=target[~surface, :]).mean(dim = 0).mean()
            # map_loss = loss_fn(est_latent_var, latent_var).mean(dim = 0).mean()
            
            surface_var_loss_cumulative += surface_var_loss.item()
            reconstruction_loss_cumulative += reconstruction_loss.item()
            map_loss_cumulative += map_loss.item()


            if minibatch:
                optimizer.zero_grad()
                total_loss = reconstruction_loss + config['lambda_map'] * map_loss
                total_loss.backward()
            else:
                total_loss += reconstruction_loss + config['lambda_map'] * map_loss
            total_loss_cumulative += total_loss.item()
            
            # Add gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            if is_scheduler_per_batch(scheduler) and minibatch:
                scheduler.step()

            total_batches += 1 * train_loader.batch_size

        if not minibatch:
            total_loss_train = total_loss_cumulative / len(train_loader)
            total_loss.backward()
        else:
            total_loss_train = total_loss_cumulative / total_batches

        surface_var_loss_train = surface_var_loss_cumulative / total_batches
        reconstruction_loss_train = reconstruction_loss_cumulative / total_batches
        map_loss_train = map_loss_cumulative / total_batches
 
        # scheduler per epoch
        if not is_scheduler_per_batch(scheduler):
            scheduler.step()
            
        # Add gradient clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        if is_val:
            # val_params = params[val_trajectories]
            total_loss_val, surface_var_loss_val, reconstruction_loss_val, map_loss_val = val(model, device, val_loader, config['lambda_map'])
            
            val_history['val_loss'].append(total_loss_val)
            val_history['surface_var_loss'].append(surface_var_loss_val)
            val_history['reconstruction_loss'].append(reconstruction_loss_val)
            val_history['map_loss'].append(map_loss_val)
        else:
            total_loss_val = total_loss_train

        # save best model
        if total_loss_val < best_loss and save_best_model:
            best_loss = total_loss_val
            if os.path.exists(f'artifacts/{model_name}'):
                save_model(model, f'artifacts/{model_name}/{model_name}_best_model.pth')
            else:
                os.makedirs(f'artifacts/{model_name}') 
                save_model(model, f'artifacts/{model_name}/{model_name}_best_model.pth')
        
        train_history['train_loss'].append(total_loss_train)
        train_history['map_loss'].append(map_loss_train)
        train_history['surface_var_loss'].append(surface_var_loss_train)
        train_history['reconstruction_loss'].append(reconstruction_loss_train)

        # Update tqdm progress bar with loss information
        if is_val:
            loop.set_postfix({
                'train_loss': f'{total_loss_train:.6f}',
                'train_map_loss': f'{map_loss_train:.6f}',
                'train_surface_var_loss': f'{surface_var_loss_train:.6f}',
                'train_reconstruction_loss': f'{reconstruction_loss_train:.6f}',
                'val_loss': f'{total_loss_val:.6f}',
                'val_map_loss': f'{map_loss_val:.6f}',
                'val_surface_var_loss': f'{surface_var_loss_val:.6f}',
                'val_reconstruction_loss': f'{reconstruction_loss_val:.6f}'
            })
        else:
            loop.set_postfix({
                'train_loss': f'{total_loss_train:.6f}',
                'train_map_loss': f'{map_loss_train:.6f}',
                'train_surface_var_loss': f'{surface_var_loss_train:.6f}',
                'train_reconstruction_loss': f'{reconstruction_loss_train:.6f}'
            })
        loop.update(1)

    if save_history:
        history_path = f'artifacts/{model_name}/{model_name}_history.pkl'
        with open(history_path, 'wb') as f:
            pickle.dump(train_history, f)
            pickle.dump(val_history, f)
    return train_history, val_history
