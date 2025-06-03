import torch
import torch_geometric
import torch.nn.functional as F
from src.utils.commons import get_config

config = get_config('configs/default.yaml')['training']

def val(model, 
        device: torch.device, 
        params: torch.Tensor,
        val_loader: torch_geometric.loader.DataLoader, 
        lambda_map: float = 1):
    model = model.to(device)
    model.eval()
    reconstruction_loss = torch.tensor(0., device=device)
    map_loss = torch.tensor(0., device=device)
    total_loss_val = 0
    reconstruction_loss_cumulative = 0
    map_loss_cumulative = 0
    total_loss_cumulative = 0
    total_batches = 0
    start_ind = 0
    
    with torch.no_grad():
        for val_batch in val_loader:
            # Move val_batch to device and ensure correct data type
            val_batch = val_batch.to(device)
            target = val_batch.x.float()
            val_batch.x = val_batch.x.float()
            
            # Get current val_batch parameters
            current_params = params[start_ind:start_ind+val_batch.batch_size]
            
            if config['amp']:
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                    out, latent_var, est_latent_var = model(val_batch, current_params)
            else:
                out, latent_var, est_latent_var = model(val_batch, current_params)
                
            start_ind += val_batch.batch_size
            
            # Calculate reconstruction loss
            reconstruction_loss = F.mse_loss(out, target)
            map_loss = F.mse_loss(est_latent_var, latent_var)
            total_loss = reconstruction_loss + lambda_map * map_loss

            reconstruction_loss_cumulative += reconstruction_loss.item()
            map_loss_cumulative += map_loss.item()
            total_loss_cumulative += total_loss.item()

            total_batches += 1 * val_loader.batch_size

        reconstruction_loss_val = reconstruction_loss_cumulative / total_batches
        map_loss_val = map_loss_cumulative / total_batches
        total_loss_val = total_loss_cumulative / total_batches

        return total_loss_val, reconstruction_loss_val, map_loss_val
