from torch_geometric.data import Data
from torch.utils.data import Dataset
from src.utils import commons
import numpy as np
import os
import h5py
import torch
import pyvista as pv


config = commons.get_config('configs/default.yaml')['config']

class GraphDataset(Dataset):
    def __init__(self, config = config, split = 'train'):
        super(GraphDataset, self).__init__()
        self.split = split
        self.config = config
        self.dataset_dir = config['dataset_dir']
        self.variable = config['variable']
        self.dim_pde = config['dim_pde']

        # load data
        self.h5_file = h5py.File(os.path.join(config['split_dir'], f'{split}.h5'), 'r')
        self.file_keys = list(self.h5_file.keys()) # remove the first 2 keys which are coordinates and edge_index
        self.file_length = len(self.file_keys)  # Set file_length to match actual number of data samples
        self.num_graphs = self.h5_file[self.file_keys[0]]['coordinates'].shape[0]
        self.surface_mask = pv.read(config["mesh_file"])["Velocity"][:, 0] == 0

        # load coordinates
        self.coordinates = self.h5_file[self.file_keys[0]]['coordinates'][:] # Convert to NumPy array
        self.num_nodes = self.coordinates.shape[0]

        # get edge attr and weights
        self.edge_list = self.h5_file[self.file_keys[0]]['edge_index'][:] # Convert to NumPy array
        self.edge_features = self.compute_edge_attr(self.edge_list)
        self.edge_weights = self.compute_edge_weights(self.edge_features)
        
        if self.variable == 'Cf':
            surface_node_indices = np.where(self.surface_mask)[0]

            self.surface_edge_mask = np.isin(self.edge_list[0], surface_node_indices) & np.isin(self.edge_list[1], surface_node_indices)

            self.surface_edge_list = self.edge_list[:, self.surface_edge_mask]
            self.surface_edge_features = self.edge_features[self.surface_edge_mask, :]
            self.surface_edge_weights = self.edge_weights[self.surface_edge_mask]

            self.coordinates = self.coordinates[self.surface_mask]


    def __del__(self):
        if hasattr(self, 'h5_file'):
            self.h5_file.close()

    def __getitem__(self, index):
        # load coordinates
        file_key = self.file_keys[index]
        params = [float(self.file_keys[index].split('_')[1]), float(self.file_keys[index].split('_')[3])]   # Assuming params are in the file name 
        params = torch.tensor(params, dtype=torch.float32).float() 
        #Load velicities
        velocities = None # Initialize velocities

        if self.dim_pde == 1:
            if self.variable == 'X':
                velocities = self.h5_file[file_key]['Ux'][:] # Convert to NumPy array
            elif self.variable == 'Y':
                velocities = self.h5_file[file_key]['Uy'][:] # Convert to NumPy array
            elif self.variable == 'Pressure':
                velocities = self.h5_file[file_key]['Pressure'][:] # Convert to NumPy array
            elif self.variable == 'Cp':
                velocities = self.h5_file[file_key]['Cp'][:] # Convert to NumPy array
                velocities = velocities[self.surface_mask]  # Apply surface mask
            elif self.variable == 'U':
                ux = self.h5_file[file_key]['Ux'][:].reshape(-1, 1)
                uy = self.h5_file[file_key]['Uy'][:].reshape(-1, 1)
                velocities = np.sqrt(ux**2 + uy**2)  # Compute magnitude of velocity
            else:
                raise ValueError(f"Unknown variable: {self.variable}")
            
        elif self.dim_pde == 2:
            if self.variable in ['X', 'Y']:
            # Load 1D arrays and reshape them to 2D before concatenating
                ux = self.h5_file[file_key]['Ux'][:].reshape(-1, 1)  # Shape: [num_nodes, 1]
                uy = self.h5_file[file_key]['Uy'][:].reshape(-1, 1)  # Shape: [num_nodes, 1]
                velocities = np.concatenate([ux, uy], axis=1)  # Shape: [num_nodes, 2]
            
            elif self.variable == 'Cf':
                velocities = self.h5_file[file_key]['Cf'][:,:] # Convert to NumPy array
                velocities = velocities[self.surface_mask, :]

        elif self.dim_pde == 3:
            ux = self.h5_file[file_key]['Ux'][:].reshape(-1, 1)  # Shape: [num_nodes, 1]
            uy = self.h5_file[file_key]['Uy'][:].reshape(-1, 1)  # Shape: [num_nodes, 1]
            pressure = self.h5_file[file_key]['Pressure'][:].reshape(-1, 1)  # Shape: [num_nodes, 1]
            velocities = np.concatenate([ux, uy, pressure], axis=1)  # Shape: [num_nodes, 3]
        
        #scale velocities
        if velocities is None:
            raise ValueError("Velocities are not loaded, cannot scale.")
        # velocities, scaler = self.scale_velocities(velocities)

        velocities = np.array(velocities)
        if velocities.ndim == 1:
            velocities = velocities.reshape(-1, 1)  # Reshape to [num_nodes, 1]

        data = Data(x = torch.tensor(velocities, dtype=torch.float32),
                    pos = torch.tensor(self.coordinates, dtype=torch.float32),
                    edge_index = torch.tensor(self.edge_list, dtype=torch.long),
                    edge_attr = torch.tensor(self.edge_features, dtype=torch.float32), 
                    edge_weight = torch.tensor(self.edge_weights, dtype=torch.float32),
                    params = params)
        
        return data
    
    def __len__(self):
        return self.file_length
    
    def scale_velocities(self, velocities):
        # Get scaling method from config
        scaling_method = self.config['scaler_name']
        
        # Ensure velocities is a NumPy array if it's not already (e.g., if it's a list)
        if not isinstance(velocities, np.ndarray):
            velocities = np.array(velocities)

        # Reshape velocities to 2D array (n_samples, 1 feature)
        if velocities.ndim == 1:
            velocities = velocities.reshape(-1, 1)
        
        if scaling_method == 'standard':
            from sklearn.preprocessing import StandardScaler # Correct import
            scaler = StandardScaler()
            velocities = scaler.fit_transform(velocities)
        elif scaling_method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler # Correct import
            scaler = MinMaxScaler()
            velocities = scaler.fit_transform(velocities)
        elif scaling_method == 'robust':
            from sklearn.preprocessing import RobustScaler # Correct import
            scaler = RobustScaler()
            velocities = scaler.fit_transform(velocities)
        else:
            raise ValueError(f"Unknown scaling method: {scaling_method}")
        return velocities, scaler
    
    def compute_edge_attr(self, edge_list):
        """edge attributes are the absolute relative position (x,y) between nodes"""
        # Assuming self.coordinates is now a NumPy array
        edge_attr = np.abs(self.coordinates[edge_list[1]] - self.coordinates[edge_list[0]])
        return torch.tensor(edge_attr, dtype=torch.float32) # Convert to tensor for consistency if needed later
    
    def compute_edge_weights(self, edge_attr):
        """edge weights are the norm of the node relative position"""
        # edge_attr is expected to be a tensor or numpy array that torch.norm can handle
        if isinstance(edge_attr, np.ndarray):
            edge_attr = torch.tensor(edge_attr)
        edge_weights = torch.norm(edge_attr, dim=1)
        return edge_weights