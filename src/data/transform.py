# src/data/transform.py

from torch_geometric.data import Data

class Transform:
    def __init__(self, mesh_file: str):
        self.mesh_file = mesh_file

    def load_mesh(self):
        pass

