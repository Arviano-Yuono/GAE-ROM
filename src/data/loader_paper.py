from torch_geometric.data import Data
from torch.utils.data import Dataset
from src.utils import commons
import numpy as np
import os
import h5py
import torch

config = commons.get_config('configs/default.yaml')['config']

class GraphDatasetPaper(Dataset):
    def __init__(self, config = config, split = 'train'):
        super(GraphDatasetPaper, self).__init__()
        self.split = split
        self.config = config
        self.dataset_dir = config['dataset_dir']
        self.variable = config['variable']
        self.dim_pde = config['dim_pde']

        # load data
        self.h5_file = h5py.File(os.path.join(config['split_dir'], f'{split}.h5'), 'r')
        self.file_keys = list(self.h5_file.keys()) # remove the first 2 keys which are coordinates and edge_index
        self.file_length = len(self.file_keys)  # Set file_length to match actual number of data samples
        self.file_index = sorted([int(key.split('_')[1]) for key in self.file_keys])
        self.num_graphs = self.h5_file[self.file_keys[0]]['coordinates'].shape[0]
        surface_mask = self.h5_file[self.file_keys[0]]['Ux'][:] == 0  # Assuming surface_mask is the same for all files
        self.surface_mask = surface_mask

    def __del__(self):
        if hasattr(self, 'h5_file'):
            self.h5_file.close()

    def __getitem__(self, index):
        # load coordinates
        file_key = self.file_keys[0].split('_')[0] + '_' + str(self.file_index[index])
        # print(f"Loading coordinates for {file_key}")
        self.coordinates = self.h5_file[file_key]['coordinates'][:] # Convert to NumPy array
        self.num_nodes = self.coordinates.shape[0]

        # get edge attr and weights
        self.edge_list = self.h5_file[file_key]['edge_index'][:] # Convert to NumPy array
        self.edge_features = self.compute_edge_attr(self.edge_list)
        self.edge_weights = self.compute_edge_weights(self.edge_features)
        #Load velicities
        # print(f"Loading velocities for index {index} out of {len(self.file_keys)}")
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
            elif self.variable == 'Cf':
                velocities = self.h5_file[file_key]['Cf'][:] # Convert to NumPy array
            else:
                raise ValueError(f"Unknown variable: {self.variable}")
        elif self.dim_pde == 2:
            print("dim_pde = 2 is not yet implemented")
            return None, None
        
        #scale velocities
        if velocities is None:
            raise ValueError("Velocities are not loaded, cannot scale.")
        # velocities, scaler = self.scale_velocities(velocities)

        data = Data(x = torch.tensor(velocities, dtype=torch.float64).float(),
                    pos = torch.tensor(self.coordinates, dtype=torch.float64),
                    edge_index = torch.tensor(self.edge_list, dtype=torch.long),
                    edge_attr = torch.tensor(self.edge_features, dtype=torch.float64), 
                    edge_weight = torch.tensor(self.edge_weights, dtype=torch.float64))
        
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
        return torch.tensor(edge_attr, dtype=torch.float64) # Convert to tensor for consistency if needed later
    
    def compute_edge_weights(self, edge_attr):
        """edge weights are the norm of the node relative position"""
        # edge_attr is expected to be a tensor or numpy array that torch.norm can handle
        if isinstance(edge_attr, np.ndarray):
            edge_attr = torch.tensor(edge_attr)
        edge_weights = torch.norm(edge_attr, dim=1)
        return edge_weights