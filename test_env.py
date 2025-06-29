import torch
import torch_geometric
import h5py
import numpy as np
import pyvista
import matplotlib.pyplot as plt
import tqdm
import yaml
import sklearn

print(f"torch version: {torch.__version__}"
      f" torch_geometric version: {torch_geometric.__version__}"
      f" is cuda available: {torch.cuda.is_available()}")

data = torch_geometric.data.Data(
    x=torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),
    edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
)

data = data.to('cuda')

print(f"graoh data: {data}")