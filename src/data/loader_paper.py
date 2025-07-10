from itertools import product
from torch_geometric.data import Data
from torch.utils.data import Dataset
from src.utils import commons
import numpy as np
import os
import h5py
import torch
import pyvista as pv
import scipy.io

config = commons.get_config('configs/default.yaml')['config']

class GraphDataset(Dataset):
    def __init__(self, config = config, split = 'train'):
        super(GraphDataset, self).__init__()
        self.split = split
        self.config = config
        self.dataset_dir = config['dataset_dir']
        self.variable = config['variable']
        self.dim_pde = config['dim_pde']
        self.with_edge_features = config.get('with_edge_features', True)

        # load data
        self.mat_file = h5py.File(os.path.join(config['split_dir'], f'{split}.h5'), 'r')
        self.file_keys = list(self.mat_file.keys()) # remove the first 2 keys which are coordinates and edge_index
        self.file_length = len(self.file_keys)  # Set file_length to match actual number of data samples
        self.num_graphs = self.mat_file[self.file_keys[0]]['coordinates'].shape[0]
            
        # load coordinates
        # Load coordinates directly - they should already be in [num_nodes, 2] format
        self.coordinates = self.mat_file[self.file_keys[0]]['coordinates'][:]  # Shape: [num_nodes, 2]
        self.num_nodes = self.coordinates.shape[0]
        # get edge attr and weights
        # self.edge_list = self.mat_file['edge_index'][:].astype(int).reshape(2,-1) - 1 # Convert to NumPy array
        # self.edge_features = torch.zeros(self.edge_list.shape[1], 1)
        # self.edge_weights = torch.ones(self.edge_list.shape[1])
        # self.surface_mask = self.mat_file[self.file_keys[0]]['Ux'][:, 0] == 0
        # self.log_scaled_distannce = self.log_scaling_distance(self.compute_implicit_distance(fluid_coords=self.coordinates,
                                                                                    #    surface_coords=self.coordinates[self.surface_mask, :2]))


    def __del__(self):
        if hasattr(self, 'mat_file'):
            self.mat_file.close()

    def __getitem__(self, index):
        # load coordinates
        file_key = self.file_keys[index]
        # Extract mu_1 and mu_2 from the group name (mu1_XX_mu2_YY format)
        group_name = self.file_keys[index]
        # if 'mu1_' in group_name and 'mu2_' in group_name:
        #     # Parse mu1_XX_mu2_YY format
        #     parts = group_name.split('_')
        #     mu1_idx = parts.index('mu1') + 1
        #     mu2_idx = parts.index('mu2') + 1
        #     mu1_val = float(parts[mu1_idx].replace('p', '.'))  # Convert '1p500' to 1.500
        #     mu2_val = float(parts[mu2_idx].replace('p', '.'))  # Convert '2p000' to 2.000
        #     params = [mu1_val, mu2_val]
        # else:
        #     # Fallback to old format if needed
        #     params = [float(self.file_keys[index].split('_')[1]), float(self.file_keys[index].split('_')[3])]
        params = self.mat_file[file_key]['parameters'][:]
        params = torch.tensor(params, dtype=torch.float32).float() 
        edge_list = self.mat_file[file_key]['edge_index'][:].astype(int).reshape(2,-1) # Convert to NumPy array
        #Load velicities
        features = None # Initialize features
        target = None  # Initialize target

        if self.dim_pde == 1:
            if self.variable == 'VX':
                ux = self.mat_file[file_key]['Ux'][:].reshape(-1, 1) # Convert to NumPy array
                implicit_distance = self.log_scaled_distannce
                features = np.concatenate([ux.reshape(-1, 1), implicit_distance.reshape(-1, 1)], axis=1)
                target = ux
            elif self.variable == 'VY':
                uy = self.mat_file[file_key]['Uy'][:].reshape(-1, 1) # Convert to NumPy array
                implicit_distance = self.log_scaled_distannce
                features = np.concatenate([uy.reshape(-1, 1), implicit_distance.reshape(-1, 1)], axis=1)
                target = uy
            elif self.variable == 'Pressure':
                p = self.mat_file[file_key]['Pressure'][:].reshape(-1, 1) # Convert to NumPy array
                implicit_distance = self.log_scaled_distannce
                features = np.concatenate([p.reshape(-1, 1), implicit_distance.reshape(-1, 1)], axis=1)  # Concatenate pressure and distance
                target = p
            elif self.variable == 'Cp':
                features = self.mat_file[file_key]['Cp'][:] # Convert to NumPy array
                features = features[self.surface_mask]  # Apply surface mask
            elif self.variable == 'U':
                ux = self.mat_file[file_key]['Ux'][:].reshape(-1, 1)
                uy = self.mat_file[file_key]['Uy'][:].reshape(-1, 1)
                features = np.sqrt(ux**2 + uy**2)  # Compute magnitude of velocity
            else:
                raise ValueError(f"Unknown variable: {self.variable}")

        elif self.variable == 're_x':
            ux = self.mat_file[file_key]['Ux'][:].reshape(-1, 1)
            # implicit_distance = self.log_scaled_distannce.reshape(-1, 1)
            # Scale mu_1 and mu_2 to [0, 1] range based on their expected ranges
            # self.surface_mask = self.mat_file[file_key]['Ux'][:, 0] == 0
            mu1_norm = (float(params[0].item()) - 0.5) / (2.0 - 0.5)  # mu1 range: [0.5, 2.0] → [0, 1]
            mu2_norm = (float(params[1].item()) - 0.5) / (2.0 - 0.5)  # mu2 range: [0.5, 2.0] → [0, 1]
            mu1 = mu1_norm * np.ones(ux.shape)  # params[0] is mu_1
            mu2 = mu2_norm * np.ones(ux.shape)  # params[1] is mu_2
            x = self.coordinates[:, 0].reshape(-1, 1)  # Shape: [num_nodes, 1]
            y = self.coordinates[:, 1].reshape(-1, 1)  # Shape
            features = np.concatenate([x, y, mu1.reshape(-1, 1), mu2.reshape(-1, 1)], axis=1)
            target = ux  # Target is the x-velocity

        elif self.variable == 're_y':
            Uy = self.mat_file[file_key]['Uy'][:].reshape(-1, 1)
            # implicit_distance = self.log_scaled_distannce.reshape(-1, 1)
            # Scale mu_1 and mu_2 to [0, 1] range based on their expected ranges
            mu1_norm = (float(params[0].item()) - 0.5) / (2.0 - 0.5)  # mu1 range: [0.5, 2.0] → [0, 1]
            mu2_norm = (float(params[1].item()) - 0.5) / (2.0 - 0.5)  # mu2 range: [0.5, 2.0] → [0, 1]
            mu1 = mu1_norm * np.ones(Uy.shape)  # params[0] is mu_1
            mu2 = mu2_norm * np.ones(Uy.shape)  # params[1] is mu_2
            x = self.coordinates[:, 0].reshape(-1, 1)  # Shape: [num_nodes, 1]
            y = self.coordinates[:, 1].reshape(-1, 1)  # Shape
            features = np.concatenate([x, y, mu1.reshape(-1, 1), mu2.reshape(-1, 1)], axis=1)
            target = Uy  # Target is the y-velocity
        
        #scale features
        if features is None:
            raise ValueError("features are not loaded, cannot scale.")
        # features, scaler = self.scale_features(features)

        features = np.array(features)
        if features.ndim == 1:
            features = features.reshape(-1, 1)  # Reshape to [num_nodes, 1]

        if target is None:
            target = features

        data = Data(x = torch.tensor(features, dtype=torch.float32),
                    y = torch.tensor(target, dtype=torch.float32),
                    pos = torch.tensor(self.coordinates, dtype=torch.float32),
                    edge_index = torch.tensor(edge_list, dtype=torch.long),
                    params = torch.tensor(params, dtype=torch.float32))
        
        return data
    
    def __len__(self):
        return self.file_length
    
    
    def compute_implicit_distance(self, fluid_coords: np.ndarray, surface_coords: np.ndarray) -> np.ndarray:        
        """
        Compute the distance from each fluid node to the nearest surface node.

        Parameters:
        - fluid_coords: [num_nodes, 2] — coordinates of all fluid domain nodes
        - surface_coords: [num_surface_nodes, 2] — coordinates of airfoil surface nodes

        Returns:
        - distances: [num_nodes] — minimum distance from each fluid node to surface
        """
        from scipy.spatial import cKDTree

        surface_kdtree = cKDTree(surface_coords)
        distances, _ = surface_kdtree.query(fluid_coords, k=1)
        return distances
    
    @staticmethod
    def log_scaling_distance(distances, eps= 2e-6):
        log_scaled = np.log1p(distances / eps)
        log_scaled = log_scaled / log_scaled.max()
        return log_scaled
