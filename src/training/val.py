
import torch
import torch_geometric
import torch.nn.functional as F


def val(model, 
        device: torch.device, 
        val_loader: torch_geometric.loader.DataLoader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            target = batch.x.float()
            batch.x = batch.x.float()
            out, _ = model(batch)
            total_loss += F.mse_loss(out, target)
        loss = total_loss / len(val_loader.dataset)
        # print(f"Validation loss: {loss.item()}")
    return loss
