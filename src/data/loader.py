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
            
        # load coordinates
        self.coordinates = self.h5_file[self.file_keys[0]]['coordinates'][:] # Convert to NumPy array
        self.num_nodes = self.coordinates.shape[0]

        if self.variable == 'Cf':
            self.surface_mask = np.ones(self.num_graphs, dtype=bool)  # Initialize surface_mask
            self.log_scaled_distannce = None 
        else:
            self.surface_mask = pv.read(config["mesh_file"])["Velocity"][:, 0] == 0
            self.log_scaled_distannce = self.log_scaling_distance(self.compute_implicit_distance(fluid_coords=self.coordinates,
                                                                                       surface_coords=self.coordinates[self.surface_mask, :2]))

        # get edge attr and weights
        self.edge_list = self.h5_file[self.file_keys[0]]['edge_index'][:] # Convert to NumPy array
        self.edge_features = self.compute_edge_attr(self.edge_list)
        self.edge_weights = self.compute_edge_weights(self.edge_features)


    def __del__(self):
        if hasattr(self, 'h5_file'):
            self.h5_file.close()

    def __getitem__(self, index):
        # load coordinates
        file_key = self.file_keys[index]
        params = [float(self.file_keys[index].split('_')[1]), float(self.file_keys[index].split('_')[3])]   # Assuming params are in the file name 
        params = torch.tensor(params, dtype=torch.float32).float() 
        #Load velicities
        features = None # Initialize features
        target = None  # Initialize target

        if self.dim_pde == 1:
            if self.variable == 'X':
                ux = self.h5_file[file_key]['Ux'][:].reshape(-1, 1) # Convert to NumPy array
                implicit_distance = self.log_scaled_distannce
                features = np.concatenate([ux.reshape(-1, 1), implicit_distance.reshape(-1, 1)], axis=1)
                target = ux
            elif self.variable == 'Y':
                uy = self.h5_file[file_key]['Uy'][:].reshape(-1, 1) # Convert to NumPy array
                implicit_distance = self.log_scaled_distannce
                features = np.concatenate([uy.reshape(-1, 1), implicit_distance.reshape(-1, 1)], axis=1)
                target = uy
            elif self.variable == 'Pressure':
                p = self.h5_file[file_key]['Pressure'][:].reshape(-1, 1) # Convert to NumPy array
                implicit_distance = self.log_scaled_distannce
                features = np.concatenate([p.reshape(-1, 1), implicit_distance.reshape(-1, 1)], axis=1)  # Concatenate pressure and distance
                target = p
            elif self.variable == 'Cp':
                features = self.h5_file[file_key]['Cp'][:] # Convert to NumPy array
                features = features[self.surface_mask]  # Apply surface mask
            elif self.variable == 'U':
                ux = self.h5_file[file_key]['Ux'][:].reshape(-1, 1)
                uy = self.h5_file[file_key]['Uy'][:].reshape(-1, 1)
                features = np.sqrt(ux**2 + uy**2)  # Compute magnitude of velocity
            else:
                raise ValueError(f"Unknown variable: {self.variable}")
            
        elif self.dim_pde == 2:
            if self.variable in ['X', 'Y']:
            # Load 1D arrays and reshape them to 2D before concatenating
                ux = self.h5_file[file_key]['Ux'][:].reshape(-1, 1)  # Shape: [num_nodes, 1]
                uy = self.h5_file[file_key]['Uy'][:].reshape(-1, 1)  # Shape: [num_nodes, 1]
                features = np.concatenate([ux, uy], axis=1)  # Shape: [num_nodes, 2]
        
        elif self.variable == 'full':
            ux = self.h5_file[file_key]['Ux'][:].reshape(-1, 1)
            uy = self.h5_file[file_key]['Uy'][:].reshape(-1, 1)
            pressure = self.h5_file[file_key]['Pressure'][:].reshape(-1, 1)
            nu_tilde = self.h5_file[file_key]['Nu_Tilde'][:].reshape(-1, 1)  # Shape: [num_nodes, 1]
            eddie_viscosity = self.h5_file[file_key]['Eddy_Viscosity'][:].reshape(-1, 1)
            implicit_distance = self.compute_implicit_distance(fluid_coords=self.coordinates, 
                                                              surface_coords=self.coordinates[self.surface_mask, :2])  # Assuming surface_mask is a boolean 
            dist_log_scaled = self.log_scaling_distance(implicit_distance).reshape(-1,1)  # Log scaling the distances
            features = np.concatenate([ux, uy, pressure, nu_tilde, eddie_viscosity, dist_log_scaled], axis=1)  # Shape: [num_nodes, 5]
            target = np.concatenate([ux, uy, pressure], axis=1)  # Shape: [num_nodes, 3]


        elif self.variable == 'Cf':
            x = self.coordinates[:, 0].reshape(-1, 1)  # Shape: [num_nodes, 1]
            y = self.coordinates[:, 1].reshape(-1, 1)  # Shape: [num_nodes, 1]
            Y_plus = self.h5_file[file_key]['Y_Plus'][:].reshape(-1, 1)  # Shape: [num_nodes, 1]
            re = np.log10(float(params[0].item()) * np.ones(x.shape)) # Assuming params[0] is Re
            alpha = np.deg2rad(float(params[1].item()) * np.ones(x.shape)) # Assuming params[1] is alpha
            cf_x = self.h5_file[file_key]['Cf'][:, 0].reshape(-1, 1)  # Ensure cf is a 2D array
            cf_y = self.h5_file[file_key]['Cf'][:, 1].reshape(-1, 1)  # Ensure cf is a 2D array
            cp = self.h5_file[file_key]['Cp'][:].reshape(-1, 1)  # Shape: [num_nodes, 1]

            features = np.concatenate([x, y, Y_plus, re, alpha], axis=1)
            target = np.concatenate([cf_x, cf_y, cp], axis=1)
            if self.edge_list.shape[1] == 0:
                from torch_geometric.transforms import RadiusGraph
                data = Data(pos = torch.tensor(self.coordinates, dtype=torch.float32))
                transform = RadiusGraph(r=0.15, max_num_neighbors=2, loop=False)  # loop=False avoids self-loops
                self.edge_list = transform(data).edge_index.numpy()  # Convert to NumPy array
                self.edge_features = self.compute_edge_attr(self.edge_list)
                self.edge_weights = self.compute_edge_weights(self.edge_features) 

        elif self.dim_pde == 3:
            ux = self.h5_file[file_key]['Ux'][:].reshape(-1, 1)  # Shape: [num_nodes, 1]
            uy = self.h5_file[file_key]['Uy'][:].reshape(-1, 1)  # Shape: [num_nodes, 1]
            pressure = self.h5_file[file_key]['Pressure'][:].reshape(-1, 1)  # Shape: [num_nodes, 1]
            features = np.concatenate([ux, uy, pressure], axis=1)  # Shape: [num_nodes, 3]
        
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
