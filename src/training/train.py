import os
import pickle
from src.model.gae import GAE
from src.training.val import val
from src.utils.commons import get_config, save_model, is_scheduler_per_batch
import torch
import torch_geometric
import torch.nn.functional as F

import tqdm


config = get_config('configs/default.yaml')['training']

def train(model: GAE, 
          device: torch.device, 
          params: torch.Tensor,
          train_loader: torch_geometric.loader.DataLoader,
          is_val: bool = False,
          val_loader: torch_geometric.loader.DataLoader = None,
          is_tqdm: bool = True,
          single_batch: bool = False,
          save_best_model: bool = True,
          save_history: bool = True,
          start_up_epoch: int = 50,
          config = config):
    
    torch.cuda.empty_cache()
    train_trajectories = [x - 1 for x in train_loader.dataset.file_index]
    train_params = params[train_trajectories].to(device)
    num_epochs = config['epochs']
    
    model_config = model.config
    model_name = f"""{model_config['encoder']['convolution_layers']['type']}"""
    # loss
    loss_fn = config['loss']['type']
    if loss_fn == 'rmse':
        loss_fn = lambda x, y: torch.sqrt(F.mse_loss(x, y))
    else:
        try:
            loss_fn = getattr(F, loss_fn)
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
    
    train_history = dict(train_loss=[], map_loss=[], reconstruction_loss=[])
    val_history = dict(val_loss=[], map_loss=[], reconstruction_loss=[])
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
        reconstruction_loss = torch.tensor(0., device=device)
        map_loss = torch.tensor(0., device=device)
        total_loss_train = 0
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
            
            # Get current batch parameters
            current_params = train_params[start_ind:start_ind+batch.batch_size]
            
            if config['amp']:
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                    out, latent_var, est_latent_var = model(batch, current_params)
            else:
                out, latent_var, est_latent_var = model(batch, current_params)
            
            start_ind += batch.batch_size

            # Calculate losses
            reconstruction_loss = F.mse_loss(input=out, target=target)
            map_loss = F.mse_loss(est_latent_var, latent_var)
            total_loss = reconstruction_loss + config['lambda_map'] * map_loss
            
            reconstruction_loss_cumulative += reconstruction_loss.item()
            map_loss_cumulative += map_loss.item()
            total_loss_cumulative += total_loss.item()

            total_loss.backward()
            optimizer.step()
            if is_scheduler_per_batch(scheduler):
                scheduler.step()

            total_batches += 1 * train_loader.batch_size

        reconstruction_loss_train = reconstruction_loss_cumulative / total_batches
        map_loss_train = map_loss_cumulative / total_batches
        total_loss_train = total_loss_cumulative / total_batches
 
        # scheduler per epoch
        if not is_scheduler_per_batch(scheduler):
            scheduler.step()
            
        # Add gradient clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        if is_val:
            val_trajectories = [x - 1 for x in val_loader.dataset.file_index]
            val_params = params[val_trajectories]
            total_loss_val, reconstruction_loss_val, map_loss_val = val(model, device, val_params, val_loader, config['lambda_map'])
            
            val_history['val_loss'].append(total_loss_val)
            val_history['reconstruction_loss'].append(reconstruction_loss_val)
            val_history['map_loss'].append(map_loss_val)
        else:
            total_loss_val = total_loss_train

        # save best model
        if total_loss_val < best_loss and save_best_model:
            best_loss = total_loss_val
            if os.path.exists(f'artifacts/no_surface/{model_name}'):
                save_model(model, f'artifacts/no_surface/{model_name}/{model_name}_best_model_{num_epochs}.pth')
            else:
                os.makedirs(f'artifacts/no_surface/{model_name}') 
                save_model(model, f'artifacts/no_surface/{model_name}/{model_name}_best_model_{num_epochs}.pth')
        
        train_history['train_loss'].append(total_loss_train)
        train_history['map_loss'].append(map_loss_train)
        train_history['reconstruction_loss'].append(reconstruction_loss_train)

        # Update tqdm progress bar with loss information
        if is_val:
            loop.set_postfix({
                'train_loss': f'{total_loss_train:.6f}',
                'map_loss': f'{map_loss_train:.6f}',
                'reconstruction_loss': f'{reconstruction_loss_train:.6f}',
                'val_loss': f'{total_loss_val:.6f}',
                'val_reconstruction_loss': f'{reconstruction_loss_val:.6f}',
                'val_map_loss': f'{map_loss_val:.6f}'
            })
        else:
            loop.set_postfix({
                'train_loss': f'{total_loss_train:.6f}',
                'map_loss': f'{map_loss_train:.6f}',
                'reconstruction_loss': f'{reconstruction_loss_train:.6f}'
            })
        loop.update(1)

    if save_history:
        history_path = r'artifacts/surface/{model_name}/{model_name}_history_{num_epochs}.pkl'
        if not os.path.exists(os.path.dirname(history_path)):
            os.makedirs(os.path.dirname(history_path))
        with open(history_path, 'wb') as f:
            pickle.dump(train_history, f)
            pickle.dump(val_history, f)
    return train_history, val_history
