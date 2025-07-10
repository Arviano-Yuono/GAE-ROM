import os
import pickle
from src.model.gae import GAE
from src.training.val_surface_paper import val
from src.utils.commons import get_config, save_model, is_scheduler_per_batch
import torch
import torch_geometric
import torch.nn.functional as F
from src.utils.metrics import relative_error
import tqdm

def hybrid_loss(pred, target, eps=1e-3, alpha=0.5):
    mse = torch.mean((pred - target) ** 2)
    rel = torch.mean(torch.abs(pred - target) / (torch.abs(target) + eps))
    return alpha * mse + (1 - alpha) * rel


config = get_config('configs/default.yaml')['training']

def train(model: GAE, 
          device: torch.device, 
          train_loader: torch_geometric.loader.DataLoader,
          is_val: bool = False,
          val_loader: torch_geometric.loader.DataLoader = None,
          is_tqdm: bool = True,
          single_batch: bool = False,
          save_best_model: bool = True,
          save_history: bool = True,
          start_up_epoch: int = 30,
          config = config):
    
    torch.cuda.empty_cache()

    model_name = f"""{config['model_name']}"""
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

    # Do this AFTER freezing
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )

    # Move this up before scheduler setup
    num_epochs = config['epochs']

    # Scheduler
    try:
        scheduler_class = getattr(torch.optim.lr_scheduler, config['scheduler']['type'])
        if config['scheduler']['type'] == 'StepLR':
            scheduler = scheduler_class(
                optimizer, 
                step_size=config['scheduler']['step_size'], 
                gamma=config['scheduler']['gamma']
            )
        elif config['scheduler']['type'] == 'CosineAnnealingLR':
            scheduler = scheduler_class(
                optimizer, 
                T_max=config['scheduler']['T_max'], 
                eta_min=config['scheduler']['eta_min']
            )
        elif config['scheduler']['type'] == 'MultiStepLR':
            scheduler = scheduler_class(
                optimizer, 
                milestones=config['scheduler']['milestones'], 
                gamma=config['scheduler']['gamma']
            )
    except AttributeError:
        raise ValueError(f"Invalid scheduler: {config['scheduler']['type']}")

    
    train_history = dict(train_loss=[], map_loss=[], reconstruction_loss=[])
    val_history = dict(val_loss=[], map_loss=[], reconstruction_loss=[])
    best_loss = float('inf')
    loss_val = None

    # training loop
    if is_tqdm:
        loop = tqdm.tqdm(range(num_epochs))
    else:
        loop = range(num_epochs)

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
            target = batch.y.float()
            batch.x = batch.x.float()
            
            # Get current batch parameters
            current_params = batch.params.float().to(device)
            
            if config['amp']:
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                    out, latent_var, est_latent_var = model(batch, current_params)
            else:
                out, latent_var, est_latent_var = model(batch, current_params)
            
            start_ind += batch.batch_size

            # MSE losses
            reconstruction_loss = F.mse_loss(input=out, target=target, reduction='mean')

            if latent_var is None or est_latent_var is None:
                map_loss = torch.tensor(0., device=device)
            else:
                # Ensure latent_var and est_latent_var are float32
                latent_var = latent_var.float()
                est_latent_var = est_latent_var.float()
                map_loss = F.mse_loss(est_latent_var, latent_var)

            # Hybrid loss
            # reconstruction_loss = hybrid_loss(pred=out[surface_mask], target=target[surface_mask]) * lambda_surface \
            # + hybrid_loss(pred=out[~surface_mask], target=target[~surface_mask])

            # if latent_var is None or est_latent_var is None:
            #     map_loss = torch.tensor(0., device=device)
            # else:
            #     # Ensure latent_var and est_latent_var are float32
            #     latent_var = latent_var.float()
            #     est_latent_var = est_latent_var.float()
            #     map_loss = hybrid_loss(pred=est_latent_var, target=latent_var)
                
            # reconstruction_loss = relative_error(pred=out[surface_mask], target=target[surface_mask]) * lambda_surface \
            # + relative_error(pred=out[~surface_mask], target=target[~surface_mask])

            # if latent_var is None or est_latent_var is None:
            #     map_loss = torch.tensor(0., device=device)
            # else:
            #     # Ensure latent_var and est_latent_var are float32
            #     latent_var = latent_var.float()
            #     est_latent_var = est_latent_var.float()
            #     map_loss = relative_error(pred = est_latent_var, target=latent_var)
                
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
            total_loss_val, reconstruction_loss_val, map_loss_val = val(model, device, val_loader, config['lambda_map'])
            
            val_history['val_loss'].append(total_loss_val)
            val_history['reconstruction_loss'].append(reconstruction_loss_val)
            val_history['map_loss'].append(map_loss_val)
        else:
            total_loss_val = total_loss_train

        # save best model
        if total_loss_val < best_loss and save_best_model and i >= start_up_epoch:
            best_loss = total_loss_val
            model_dir = f'artifacts/paper/{model_name}'
            model_path = f'{model_dir}/{model_name}_best_model_{num_epochs}.pth'
            history_path = f'{model_dir}/{model_name}_history_{num_epochs}.pkl'
            
            # Ensure directory exists
            os.makedirs(model_dir, exist_ok=True)
            
            # Save model
            save_model(model, model_path)
            
            # Save history
            with open(history_path, 'wb') as f:
                pickle.dump(train_history, f)
                pickle.dump(val_history, f)

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
        history_path = f'artifacts/paper/{model_name}/{model_name}_history_{num_epochs}.pkl'
        if not os.path.exists(os.path.dirname(history_path)):
            os.makedirs(os.path.dirname(history_path))
        with open(history_path, 'wb') as f:
            pickle.dump(train_history, f)
            pickle.dump(val_history, f)
    return train_history, val_history
