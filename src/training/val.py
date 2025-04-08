
import torch
import torch_geometric
import torch.nn.functional as F


def val(model, 
        device: torch.device, 
        val_loader: torch_geometric.loader.DataLoader):
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            target = batch.x
            out, _ = model(batch)
            loss = F.mse_loss(out, target)
            # print(f"Validation loss: {loss.item()}")
    return loss
