import torch
from src.model.gae import GAE
from src.utils.commons import get_config, get_device
from src.data.loader_paper import GraphDatasetPaper
from torch_geometric.loader import DataLoader

# Load configuration
config = get_config('configs/paper.yaml')

# Create dataset
train_dataset = GraphDatasetPaper(config=config['config'], split='train')
device = get_device()

# Create data loader
train_loader = DataLoader(dataset=train_dataset, 
                         batch_size=1, 
                         shuffle=False,
                         num_workers=0)

# Initialize model
model = GAE(config, num_graphs=train_dataset.num_graphs).to(device)

# Test with one batch
for batch in train_loader:
    batch = batch.to(device)
    print(f"Input shape: {batch.x.shape}")
    
    try:
        with torch.no_grad():
            out, latent_var, est_latent_var = model(batch)
        print(f"Output shape: {out.shape}")
        print(f"Latent var shape: {latent_var.shape}")
        print(f"Estimated latent var shape: {est_latent_var.shape}")
        print("Model forward pass successful!")
        break
    except Exception as e:
        print(f"Error: {e}")
        break 