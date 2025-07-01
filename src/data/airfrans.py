from torch_geometric.data import Data
from torch.utils.data import Dataset
from src.utils import commons
import numpy as np
import torch_geometric.nn as nng
import torch
from torch_geometric.datasets import AirfRANS as af
import os.path as osp

config = commons.get_config('configs/default.yaml')['config']

class DatasetAirfRANS(Dataset):
    def __init__(self, 
                 split = 'train', 
                 task = 'aoa', 
                 root = 'dataset/AirFRans_processed/processed'):
        super(DatasetAirfRANS, self).__init__()
        if split == 'train':
            train = True
        else:
            train = False

        self.device = commons.get_device()
        self.split = split
        self.config = config
        self.task = task
        self.train = train

        # load data
        self.dataset = torch.load(osp.join(root, f'{task}_{split}_normalized.pt'), weights_only = False)
        self.num_graphs = self.dataset[0].x.shape[0]

    def __getitem__(self, index):
        self.dataset[index].params = torch.tensor([float(self.dataset[index].name.split('_')[2]), float(self.dataset[index].name.split('_')[3])], dtype=torch.float32, device=self.device)
        self.surface_mask = self.dataset[index].surf.bool().to(self.device)
        self.dataset[index].x = self.dataset[index].x[:, :4]
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)