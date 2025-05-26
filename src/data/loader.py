# src/data/loader.py

import torch_geometric
from torch_geometric.data import Data
from torch.utils.data import Dataset
from src.utils import commons
import numpy as np
import meshio
import os
import h5py
import torch
import scipy

config = commons.get_config('configs/default.yaml')['config']

class GraphDataset(Dataset):
    def __init__(self, config = config, split = 'train'):
        super(GraphDataset, self).__init__()
        self.split = split
        self.config = config
        self.dataset_dir = config['dataset_dir']
        self.variable = config['variable']
        self.mesh_file = config['mesh_file']
        self.variable = config['variable']
        self.dim_pde = config['dim_pde']

        # load mesh
        self.points, self.triangles, self.areas = self.transform_mesh()
        self.cell_coordinates = self.compute_cell_center_coordinates()
        if self.config['preprocessing']['with_edge_features']:
            self.edge_list = self.calculate_edge_list()
            # self.edge_list, self.edge_weights, self.edge_features = self.compute_edge_features()
            self.edge_features = self.compute_edge_attr(self.edge_list)
            self.edge_weights = self.compute_edge_weights(self.edge_features)
        else:
            self.edge_list = self.calculate_edge_list()
            self.edge_weights = np.ones(self.edge_list.shape[1], dtype=np.float32)
            self.edge_features = torch.zeros((0, 3), dtype=torch.float32)
        self.num_nodes = self.triangles.shape[0]

        # load data
        self.h5_file = h5py.File(os.path.join(config['split_dir'], f'{split}.h5'), 'r')
        self.file_length = len(self.h5_file)
        self.file_keys = list(self.h5_file.keys())

    def __del__(self):
        if hasattr(self, 'h5_file'):
            self.h5_file.close()

    def __getitem__(self, index):
        if self.dim_pde == 1:
            if self.variable == 'X':
                velocities = self.h5_file[self.file_keys[index]]['Ux'][:]
            elif self.variable == 'Y':
                velocities = self.h5_file[self.file_keys[index]]['Uy'][:]
        elif self.dim_pde == 2:
            velocities = self.h5_file[self.file_keys[index]]['velocity'][:]
        
        if 'velocities' not in locals(): # Check if velocities was assigned
            raise ValueError(f"Velocities not loaded for index {index}, variable {self.variable}, dim_pde {self.dim_pde}")

        cell_velocities = self.compute_cell_center_velocities(velocities)
        cell_velocities = self.normalize_velocities(cell_velocities)

        if self.config['preprocessing']['with_edge_features']:
            data = Data(x = torch.tensor(cell_velocities, dtype=torch.float32),
                        pos = torch.tensor(self.cell_coordinates, dtype=torch.float32),
                        edge_index = torch.tensor(self.edge_list, dtype=torch.long),
                        edge_attr = torch.tensor(self.edge_features, dtype=torch.float32), 
                        edge_weight = torch.tensor(self.edge_weights, dtype=torch.float32))
        else:
            data = Data(x = torch.tensor(cell_velocities, dtype=torch.float32),
                        pos = torch.tensor(self.cell_coordinates, dtype=torch.float32),
                        edge_index = torch.tensor(self.edge_list, dtype=torch.long))
        return data
    
    def __len__(self):
        return self.file_length
    
    def normalize_velocities(self, cell_velocities):
        # Get normalization method from config
        norm_method = self.config['preprocessing']['normalization_method']  # default to zscore
        
        if norm_method == 'zscore':
            # Original z-score normalization
            mean_velocities = np.mean(cell_velocities, axis=0) 
            std_velocities = np.std(cell_velocities, axis=0)
            cell_velocities[:, 0] = (cell_velocities[:, 0] - mean_velocities[0]) / std_velocities[0]
            cell_velocities[:, 1] = (cell_velocities[:, 1] - mean_velocities[1]) / std_velocities[1]
            
        elif norm_method == 'magnitude':
            # Velocity magnitude normalization
            velocity_magnitude = np.sqrt(np.sum(cell_velocities**2, axis=1))
            max_magnitude = np.max(velocity_magnitude)
            cell_velocities = cell_velocities / max_magnitude

        elif norm_method == 'robust':
            # Robust normalization using IQR
            q1 = np.percentile(cell_velocities, 25, axis=0)
            q3 = np.percentile(cell_velocities, 75, axis=0)
            iqr = q3 - q1
            median_velocities = np.median(cell_velocities, axis=0)
            cell_velocities = (cell_velocities - median_velocities) / iqr
            
        else:
            raise ValueError(f"Unknown normalization method: {norm_method}")
            
        return cell_velocities
    
    def load_mesh(self):
        if self.mesh_file.endswith('.vtk'):
            mesh = meshio.read(self.mesh_file, file_format='vtk')
        elif self.mesh_file.endswith('.su2'):
            mesh = meshio.read(self.mesh_file, file_format='su2')
        else:
            raise ValueError(f"Unknown mesh file format: {self.mesh_file}")
        return mesh
    
    def calculate_edge_list(self):
        # calculate the edge list from the mesh triangles
        edge_to_faces = self.build_edge_to_faces(self.triangles)
        edge_list = []
        for edge, face_indices in edge_to_faces.items():
            if len(face_indices) == 2:
                edge_list.append(face_indices)
        
        return np.array(edge_list).T  # shape: (2, num_edges)

    def transform_mesh(self):
        self.mesh = self.load_mesh()
        points = self.mesh.points
        triangles = self.mesh.cells[2].data
        # Convert 1-based indices to 0-based indices
        triangles = triangles - 1

        areas = GraphDataset.calculate_area_of_triangles(points, triangles) 
        return points, triangles, areas

    def compute_edge_normals(self, points, triangles, edge_to_faces):
        edge_normals = []
        for edge, face_indices in edge_to_faces.items():
            normals = []
            p1 = points[edge[0]][:2]
            p2 = points[edge[1]][:2]
            edge_vec = p2 - p1
            candidate1 = np.array([-edge_vec[1], edge_vec[0]])
            candidate2 = -candidate1

            for face_idx in face_indices:
                tri = triangles[face_idx]
                tri_pts = points[tri][:, :2]
                centroid = tri_pts.mean(axis=0)
                midpoint = (p1 + p2) / 2.0
                if np.dot(candidate1, midpoint - centroid) > 0:
                    normal = candidate1
                else:
                    normal = candidate2
                # Normalize
                norm = np.linalg.norm(normal)
                if norm != 0:
                    normal = normal / norm
                normals.append(normal)
            
            normal_avg = np.mean(normals, axis=0)
            norm = np.linalg.norm(normal_avg)
            if norm != 0:
                normal_avg = normal_avg / norm
            edge_normals.append(normal_avg)
        
        return np.stack(edge_normals, axis=0)
    
    def compute_edge_attr(self, edge_list):
        """edge attributes are the absolute relative position (x,y) between nodes"""
        edge_attr = torch.abs(torch.tensor(self.cell_coordinates[edge_list[1]], dtype=torch.float32) - torch.tensor(self.cell_coordinates[edge_list[0]], dtype=torch.float32))
        return edge_attr
    
    def compute_edge_weights(self, edge_attr):
        """edge weights are the norm of the node relative position"""
        edge_weights = torch.norm(edge_attr, dim=1)
        return edge_weights
    
    def compute_edge_features(self):
        edge_list = self.calculate_edge_list()
        edge_weights = np.ones(edge_list.shape[1], dtype=np.float32)
        edge_distances = self.compute_edge_distances(edge_list)
        edge_features = torch.tensor(edge_distances, dtype=torch.float32)
        
        return edge_list, edge_weights, edge_features
    
    def compute_edge_distances(self, edge_list):
        edge_distances = []
        for edge in edge_list.T:
            dist = np.linalg.norm(self.cell_coordinates[edge[0]] - self.cell_coordinates[edge[1]])
            # print(dist)
            edge_distances.append(dist)
        return np.array(edge_distances)
    
    @staticmethod
    def compute_edge_distances_static(cell_coordinates, edge_list):
        edge_distances = []
        for edge in edge_list.T:
            dist = np.linalg.norm(cell_coordinates[edge[0]] - cell_coordinates[edge[1]])
            edge_distances.append(dist)
        return np.array(edge_distances)
    
    def get_data(self):
        points, triangles, areas = self.transform_mesh()
        edge_list, edge_weights, edge_features = self.compute_edge_features()
        return points, triangles, areas, edge_list, edge_weights, edge_features
    
    def compute_cell_center_coordinates(self):
        cell_coords = []
        for tri in self.triangles:
            tri_pts = self.points[tri, :2]
            centroid = np.mean(tri_pts, axis=0)
            cell_coords.append(centroid)
        return np.array(cell_coords)
    
    def compute_cell_center_velocities(self, velocity_field):
        cell_velocities = []
        valid_triangles = []
        
        # First, validate all triangles
        for tri in self.triangles:
            if np.any(tri >= len(velocity_field)):
                print(f"Warning: Triangle indices {tri} are out of bounds for velocity field of size {len(velocity_field)}")
                continue
            valid_triangles.append(tri)
        
        if not valid_triangles:
            raise ValueError(f"No valid triangles found. Velocity field size: {len(velocity_field)}, Max triangle index: {np.max(self.triangles)}")
            
        # Process valid triangles
        for tri in valid_triangles:
            tri_node_vels = velocity_field[tri]
            tri_avg_vel = np.mean(tri_node_vels, axis=0) 
            cell_velocities.append(tri_avg_vel)
            
        return np.array(cell_velocities)

    @staticmethod
    def calculate_area(points):
        # print(f"Points: {points}")
        x1, y1, _ = points[0]
        x2, y2, _ = points[1]
        x3, y3, _ = points[2]
        return 0.5 * np.abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

    @staticmethod
    def calculate_area_of_triangles(points, triangles):
        areas = []
        for triangle in triangles:
            areas.append(GraphDataset.calculate_area(points[triangle]))
        return areas



class LoadDatasetGCA(torch_geometric.data.Dataset):
    """
    A custom dataset class which loads data from a .mat file using scipy.io.loadmat.

    data_mat : scipy.io.loadmat
        The loaded data in a scipy.io.loadmat object.
    U : torch.Tensor
        The tensor representation of the specified variable from the data_mat.
    xx : torch.Tensor
        The tensor representation of the 'xx' key from the data_mat. Refers to X coordinates of the domain
    yy : torch.Tensor
        The tensor representation of the 'yy' key from the data_mat.Refers to Y coordinates of the domain
    zz : torch.Tensor
        The tensor representation of the 'zz' key from the data_mat.Refers to Z coordinates of the domain
    dim : Integer
        The integer dim denotes the dimensionality of the domain where the pde is posed
    T : torch.Tensor
        The tensor representation of the 'T' key from the data_mat, casted to int. Adjacency Matrix
    E : torch.Tensor
        The tensor representation of the 'E' key from the data_mat, casted to int. Connection Matrix

    __init__(self, root_dir, variable)
        Initializes the LoadDataset object by loading the data from the .mat file at the root_dir location and converting the specified variable to a tensor representation.
    """

    def __init__(self, root_dir, variable, dim_pde, n_comp):
        # Load your mat file here using scipy.io.loadmat
        self.data_mat = scipy.io.loadmat(root_dir)
        self.dim = dim_pde
        self.n_comp = n_comp
        self.xx = torch.tensor(self.data_mat['xx'])
        self.yy = torch.tensor(self.data_mat['yy'])
        self.T = torch.tensor(self.data_mat['T'].astype(int))
        self.E = torch.tensor(self.data_mat['E'].astype(int))

        if self.n_comp == 1:
            self.U = torch.tensor(self.data_mat[variable])
        elif self.n_comp == 2:
            self.VX = torch.tensor(self.data_mat['VX'])
            self.VY = torch.tensor(self.data_mat['VY'])

        if self.dim == 3:
            self.zz = torch.tensor(self.data_mat['zz'])


    def len(self):
        pass
    
    def get(self):
        pass