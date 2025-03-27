# src/data/loader.py

# load data from SU2 .vtk files
# TODO: load data from SU2 .vtk files and compile the data into the get_data() function 

from torch.utils.data import Dataset
from src.utils import commons
import numpy as np
import meshio

config = commons.get_config('configs/default.yaml')


class LoadDataset(Dataset):
    def __init__(self, dataset_dir = config['config']['dataset_dir'], 
                    variable = config['config']['variable'],
                    mesh_file = config['config']['mesh_file']):
        self.dataset_dir = dataset_dir
        self.variable = variable
        self.mesh_file = mesh_file

    def load_mesh(self):
        mesh = meshio.read(self.mesh_file)
        return mesh
    
    def transform_mesh(self):
        mesh = self.load_mesh()
        points = mesh.points
        triangles = mesh.cells[0].data
        # print(triangles)
        areas = LoadDataset.calculate_area_of_triangles(points, triangles)
        return points, triangles, areas

    def load_data(self):
        """
        Load time history variables data from SU2 .vtk files
        """
        pass

    def compute_edge_features(self):
        points, triangles, _ = self.transform_mesh()
        edge_to_faces = self.build_edge_to_faces(triangles)
        edge_list = np.array(list(edge_to_faces.keys())).T  # shape: (2, num_edges)

        edge_features = []
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
            edge_features.append(normal_avg)

        edge_features = np.stack(edge_features, axis=0)
        return edge_list, edge_features

    def get_data(self):
        points, triangles, areas = self.transform_mesh()
        edge_list, edge_features = self.compute_edge_features()
        return points, triangles, areas, edge_list, edge_features

    @staticmethod
    def calculate_area(points):
        a = np.sqrt((points[0][0] - points[1][0])**2 + (points[0][1] - points[1][1])**2)
        b = np.sqrt((points[1][0] - points[2][0])**2 + (points[1][1] - points[2][1])**2)
        c = np.sqrt((points[2][0] - points[0][0])**2 + (points[2][1] - points[0][1])**2)
        return a + b + c

    @staticmethod
    def calculate_area_of_triangles(points, triangles):
        areas = []
        for triangle in triangles:
            areas.append(LoadDataset.calculate_area(points[triangle]))
        return areas

    def build_edge_to_faces(self, triangles):
        edge_to_faces = {}
        for face_idx, tri in enumerate(triangles):
            for i in range(3):
                edge = (tri[i], tri[(i+1) % 3])
                edge_sorted = tuple(sorted(edge))
                if edge_sorted not in edge_to_faces:
                    edge_to_faces[edge_sorted] = []
                edge_to_faces[edge_sorted].append(face_idx)
        return edge_to_faces