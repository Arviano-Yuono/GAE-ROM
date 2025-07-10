from torch_geometric.data import Data
from torch.utils.data import Dataset
from src.utils import commons
import numpy as np
import os
import h5py
import torch
import pyvista as pv

class GraphDataset(Dataset):
    def __init__(self, dataset_dir, split = 'train'):
        super(GraphDataset, self).__init__()
        self.split = split
        self.dataset_dir = dataset_dir

        # load data
        self.pt_file = torch.load(os.path.join(dataset_dir, f'{split}.pt'), weights_only=False)
        self.file_keys = []
        self.params_list = []
        for data in self.pt_file:
            self.file_keys.append(f"Re_{data.params[0]}_alpha_{data.params[1]}")
            self.params_list.append(data.params.cpu().numpy())
            data = data.to('cuda:0')
        self.file_length = len(self.pt_file)  # Set file_length to match actual number of data samples
        self.num_graphs = self.pt_file[0].x.shape[0]
            
        self.num_nodes = self.pt_file[0].x.shape[0]
        self.surface_mask = self.pt_file[0].surface_mask.cpu().numpy()
        self.log_scaled_distannce = self.pt_file[0].x[:,2]

    def __getitem__(self, index):
        return self.pt_file[index]
    
    def __len__(self):
        return self.file_length
