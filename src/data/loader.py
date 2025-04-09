# src/data/loader.py

from torch_geometric.data import Data
from torch.utils.data import Dataset
from src.utils import commons
import numpy as np
import meshio
import os
import h5py
import torch

config = commons.get_config('configs/default.yaml')['config']

class GraphDataset(Dataset):
    def __init__(self, config = config, split = 'train'):
        super(GraphDataset, self).__init__()
        self.config = config
        self.dataset_dir = config['dataset_dir']
        self.variable = config['variable']
        self.mesh_file = config['mesh_file']
        self.split = split
        
        # load mesh
        self.points, self.triangles, self.areas = self.transform_mesh()
        self.cell_coordinates = self.compute_cell_center_coordinates()
        if self.config['with_edge_features']:
            self.edge_list = self.calculate_edge_list()
            self.edge_list, self.edge_weights, self.edge_features = self.compute_edge_features()
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
        velocities = self.h5_file[self.file_keys[index]]['velocity'][:]
        cell_velocities = self.compute_cell_center_velocities(velocities)
        cell_velocities = self.normalize_velocities(cell_velocities)

        if self.config['with_edge_features']:
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
        norm_method = self.config.get('normalization_method', 'zscore')  # default to zscore
        
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
            
        else:
            raise ValueError(f"Unknown normalization method: {norm_method}")
            
        return cell_velocities
    
    def load_mesh(self):
        mesh = meshio.read(self.mesh_file, file_format='vtk')
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

    def compute_edge_features(self):
        self.points, self.triangles, self.areas = self.transform_mesh()
        edge_to_faces = self.build_edge_to_faces(self.triangles)
        edge_list = self.calculate_edge_list()
        # print(f"Edge list: {edge_list}")
        
        if edge_list.size == 0:
            print("Warning: No edges found in the mesh!")
            # Return empty tensors with correct shapes
            edge_list = torch.zeros((2, 0), dtype=torch.long)
            edge_weights = torch.zeros(0, dtype=torch.float32)
            edge_features = torch.zeros((0, 3), dtype=torch.float32)  # 2 for normal + 1 for area
            return edge_list, edge_weights, edge_features

        edge_weights = np.ones(edge_list.shape[1], dtype=np.float32)

        # Compute edge distances
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

    def build_edge_to_faces(self, triangles):
        # print('this is the edge to faces')
        edge_to_faces = {}
        for face_idx, tri in enumerate(triangles):
            for i in range(3):
                # Create edge with sorted vertices to ensure consistent edge representation
                node1, node2 = sorted([tri[i], tri[(i+1) % 3]])
                edge = (node1, node2)
                if edge not in edge_to_faces:
                    edge_to_faces[edge] = []
                edge_to_faces[edge].append(face_idx)
        return edge_to_faces