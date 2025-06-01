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
          config = config):
    
    torch.cuda.empty_cache()
    train_trajectories = train_loader.dataset.file_index
    train_params = params[train_trajectories]

    model_config = model.config
    model_name = f"""{model_config['encoder']['convolution_layers']['type']}_\
        {model_config['encoder']['pool']['type']}_\
            {model_config['encoder']['pool']['ratio']}_\
                {model_config['encoder']['pool']['is_pooling']}"""
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
    train_history = dict(train_loss=[], val_loss=[], map_loss=[])
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
        loss_train = 0
        train_loss_cumulative = 0
        map_loss_cumulative = 0
        total_batches = 0
        start_ind = 0
        for batch in train_loader:
            optimizer.zero_grad()
            if config['amp']:
                with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                    batch = batch.to(device)
                    target = batch.x.float()
                    batch.x = batch.x.float()
                    out, latent_var, est_latent_var = model(batch, train_params[start_ind:start_ind+batch.batch_size])
                    start_ind += batch.batch_size
                out = out.to(torch.float64)
            else:
                batch = batch.to(device)
                target = batch.x.float()
                batch.x = batch.x.float()
                out, latent_var, est_latent_var = model(batch, train_params[start_ind:start_ind+batch.batch_size])
                start_ind += batch.batch_size

            # Calculate reconstruction loss
            reconstruction_loss = F.mse_loss(input=out, target=target)
            # Calculate mapping loss (MSE between estimated and actual latent variables)
            if latent_var is not None and est_latent_var is not None:
                map_loss = F.mse_loss(est_latent_var, latent_var)
                # Combine losses with weight factor
                total_loss = reconstruction_loss + config['lambda_map'] * map_loss
            else:
                total_loss = reconstruction_loss

            total_loss.backward()
            optimizer.step()
            if is_scheduler_per_batch(scheduler):
                scheduler.step()
            train_loss_cumulative += reconstruction_loss.item()
            if latent_var is not None and est_latent_var is not None:
                map_loss_cumulative += map_loss.item()
            total_batches += 1 * config['batch_size']

        loss_train = torch.tensor(train_loss_cumulative / total_batches, device=device)
        if latent_var is not None and est_latent_var is not None:
            map_loss_avg = torch.tensor(map_loss_cumulative / total_batches, device=device)
        else:
            map_loss_avg = torch.tensor(0., device=device)

        # scheduler per epoch
        if not is_scheduler_per_batch(scheduler):
            scheduler.step()
            
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        if is_val:
            val_trajectories = val_loader.dataset.file_index
            val_params = params[val_trajectories]
            loss_val = val(model, device, val_params, val_loader, config['lambda_map'])
            train_history['val_loss'].append(loss_val.item())
            # save best model
            if loss_val.item() < best_loss and save_best_model:
                best_loss = loss_val.item()
                save_model(model, f'artifacts/{model_name}_best_model.pth')
        else:
            if loss_train.item() < best_loss and save_best_model:
                best_loss = loss_train.item()
                save_model(model, f'artifacts/{model_name}_best_model.pth')
        
        train_history['train_loss'].append(loss_train.item())
        train_history['map_loss'].append(map_loss_avg.item())

        if i % config['print_train'] == 0:
            if is_val:
                print(f"Epoch {i+1}/{config['epochs']}, train_loss: {loss_train.item():.6f}, map_loss: {map_loss_avg.item():.6f}, val_loss: {loss_val.item():.6f}")
            else:
                print(f"Epoch {i+1}/{config['epochs']}, train_loss: {loss_train.item():.6f}, map_loss: {map_loss_avg.item():.6f}")
    
    return train_history
