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
        self.dataset_dir = config['dataset_dir']
        self.variable = config['variable']
        self.mesh_file = config['mesh_file']
        self.split = split
        
        # load mesh
        self.points, self.triangles, self.areas = self.transform_mesh()
        self.cell_coordinates = self.compute_cell_center_coordinates()
        self.edge_list, self.edge_weights, self.edge_features = self.compute_edge_features()
        
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
        # print(f"Cell velocities shape: {cell_velocities.shape}")
        # print(f"Cell coordinates shape: {cell_coordinates.shape}")
        # print(f"Edge list shape: {self.edge_list.shape}")
        # print(f"Edge features shape: {self.edge_features.shape}")
        # print(f"Edge weights shape: {self.edge_weights.shape}")
        data = Data(x = torch.tensor(cell_velocities, dtype=torch.float32),
                    pos = torch.tensor(self.cell_coordinates, dtype=torch.float32),
                    edge_index = torch.tensor(self.edge_list, dtype=torch.long), 
                    edge_attr = torch.tensor(self.edge_features, dtype=torch.float32),
                    edge_weight = torch.tensor(self.edge_weights, dtype=torch.float32))
        return data
    
    def __len__(self):
        return self.file_length
    
    def load_mesh(self):
        mesh = meshio.read(self.mesh_file, file_format='vtk')
        return mesh
    
    def calculate_edge_list(self):
        # calculate the edge list from the mesh triangles
        mesh = self.load_mesh()
        triangles = mesh.cells[2].data
        
        # Use build_edge_to_faces to get the mapping
        edge_to_faces = self.build_edge_to_faces(triangles)
        # print(f"Edge to faces: {edge_to_faces}")
        # Create edges between triangles sharing faces
        edge_list = []
        # print the max min of the edge_to_faces
        # print(f"Max: {max(edge_to_faces.keys())}")
        # print(f"Min: {min(edge_to_faces.keys())}")
        # print(f"Edge to faces items: {edge_to_faces.items()}")
        for edge, face_indices in edge_to_faces.items():
            # If a face is shared by exactly two triangles, create an edge between them
            # print(f"Face indices: {face_indices}")
            # print(f"Edge: {edge}")
            if len(face_indices) == 2:
                edge_list.append(face_indices)
        
        return np.array(edge_list).T  # shape: (2, num_edges)

    def transform_mesh(self):
        mesh = self.load_mesh()
        points = mesh.points
        triangles = mesh.cells[2].data
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
        
        num_triangles = len(self.triangles)
        # print(f"Number of triangles: {num_triangles}")
        
        # Ensure edge indices are within bounds
        # valid_edges = (edge_list[0] < num_triangles) & (edge_list[1] < num_triangles)
        
        # edge_list = edge_list[:, valid_edges]

        edge_weights = np.ones(edge_list.shape[1], dtype=np.float32)
        # print(f"Edge weights shape: {edge_weights.shape}")

        # Compute edge normals
        # edge_normals = self.compute_edge_normals(self.points, self.triangles, edge_to_faces)
        # print(f"Edge normals: {edge_normals.shape}")

        # Compute edge normalized distances
        edge_distances = self.compute_edge_distances(edge_list)
        edge_features = torch.tensor(edge_distances, dtype=torch.float32)
        # print(f"Edge distances: {edge_distances.shape}")

        # Compute edge areas
        # edge_areas = []
        # for edge, face_indices in edge_to_faces.items():
        #     edge_area = np.mean([self.areas[face_idx] for face_idx in face_indices])
        #     edge_areas.append(edge_area)
        
        # edge_areas = np.array(edge_areas)
        # edge_areas = np.expand_dims(edge_areas, axis=1)
        # print(f"Edge areas shape: {edge_areas.shape}")
        # print(f"Edge normals shape: {edge_normals.shape}")
        # edge_features = np.concatenate([edge_distances], axis=1)
        
        # Filter edge features to match valid edges
        # edge_features = edge_features[valid_edges]
        # print(f"Edge features shape after filtering: {edge_features.shape}")
        
        # Convert to torch tensors and ensure correct types
        edge_list = torch.tensor(edge_list, dtype=torch.long)
        edge_weights = torch.tensor(edge_weights, dtype=torch.float32)
        # edge_features = torch.tensor(edge_features, dtype=torch.float32)
        
        
        return edge_list, edge_weights, edge_features

    def compute_edge_distances(self, edge_list):
        edge_distances = []
        for edge in edge_list.T:
            dist = np.linalg.norm(self.cell_coordinates[edge[0]] - self.cell_coordinates[edge[1]])
            # print(dist)
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