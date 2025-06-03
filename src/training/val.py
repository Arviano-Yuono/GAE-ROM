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
    model.eval()
    total_reconstruction_loss = 0
    total_map_loss = 0
    start_ind = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device and ensure correct data type
            batch = batch.to(device)
            target = batch.x.float()
            batch.x = batch.x.float()
            
            # Get current batch parameters
            current_params = params[start_ind:start_ind+batch.batch_size]
            
            if config['amp']:
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                    out, latent_var, est_latent_var = model(batch, current_params)
            else:
                out, latent_var, est_latent_var = model(batch, current_params)
                
            start_ind += batch.batch_size
            
            # Calculate reconstruction loss
            reconstruction_loss = F.mse_loss(out, target)
            total_reconstruction_loss += reconstruction_loss

            # Calculate mapping loss if latent variables are available
            if latent_var is not None and est_latent_var is not None:
                map_loss = F.mse_loss(est_latent_var, latent_var)
                total_map_loss += map_loss

        # Average losses
        avg_reconstruction_loss = total_reconstruction_loss / len(val_loader.dataset)
        if latent_var is not None and est_latent_var is not None:
            avg_map_loss = total_map_loss / len(val_loader.dataset)
            # Combine losses with weight factor
            total_loss = avg_reconstruction_loss + lambda_map * avg_map_loss
        else:
            total_loss = avg_reconstruction_loss

    return total_loss
