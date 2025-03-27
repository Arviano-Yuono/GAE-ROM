import meshio
import numpy as np
import torch
from torch_geometric.data import Data

# Load the SU2 mesh using meshio.
mesh = meshio.read("mesh_cylinder_lam.su2")  # Update with your actual file path

# Extract points (node positions)
points = mesh.points  # shape: (num_nodes, dim)

# Extract triangle cells (assuming the interior elements are triangles)
triangles = None
for cell in mesh.cells:
    if cell.type == "triangle":
        triangles = cell.data
        break
if triangles is None:
    raise ValueError("No triangle cells found in the mesh.")

num_nodes = points.shape[0]

# Compute nodal areas.
# For each triangle, compute the area (using the 2D formula) and then add one-third of it to each node.
node_area = np.zeros(num_nodes)
triangle_areas = []
for tri in triangles:
    # Get the triangle's vertex coordinates (assuming the mesh is 2D or that we use only the first two dimensions)
    pts = points[tri][:, :2]
    # Triangle area: 0.5 * |(x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)|
    area = 0.5 * np.abs((pts[1, 0] - pts[0, 0])*(pts[2, 1] - pts[0, 1]) - 
                         (pts[2, 0] - pts[0, 0])*(pts[1, 1] - pts[0, 1]))
    triangle_areas.append(area)
    for i in tri:
        node_area[i] += area / 3.0
triangle_areas = np.array(triangle_areas)

# Build a dictionary mapping each unique edge (stored in sorted order) to the list of adjacent triangle indices.
edge_to_faces = {}
for face_idx, tri in enumerate(triangles):
    # List the 3 edges (each as a tuple of node indices)
    for i in range(3):
        edge = (tri[i], tri[(i+1) % 3])
        edge_sorted = tuple(sorted(edge))
        if edge_sorted not in edge_to_faces:
            edge_to_faces[edge_sorted] = []
        edge_to_faces[edge_sorted].append(face_idx)

# Convert the edge keys into an edge_index array (2 x num_edges)
edge_list = np.array(list(edge_to_faces.keys())).T  # shape: (2, num_edges)

# Compute edge features: for each edge, compute the face-normal direction.
# For each adjacent face, we compute the outward normal for that edge.
edge_features = []
for edge, face_indices in edge_to_faces.items():
    normals = []
    # Get the positions for the two endpoints of the edge (use only first two dims)
    p1 = points[edge[0]][:2]
    p2 = points[edge[1]][:2]
    edge_vec = p2 - p1
    # Two candidate normals (perpendicular directions)
    candidate1 = np.array([-edge_vec[1], edge_vec[0]])
    candidate2 = -candidate1

    for face_idx in face_indices:
        tri = triangles[face_idx]
        # Get the coordinates for the triangle (first two dimensions)
        tri_pts = points[tri][:, :2]
        centroid = tri_pts.mean(axis=0)
        midpoint = (p1 + p2) / 2.0
        # Choose the candidate that points away from the triangle centroid.
        if np.dot(candidate1, midpoint - centroid) > 0:
            normal = candidate1
        else:
            normal = candidate2
        # Normalize the computed normal
        norm = np.linalg.norm(normal)
        if norm != 0:
            normal = normal / norm
        normals.append(normal)
    # Average the normals from adjacent faces (if more than one) and re-normalize.
    normal_avg = np.mean(normals, axis=0)
    norm = np.linalg.norm(normal_avg)
    if norm != 0:
        normal_avg = normal_avg / norm
    edge_features.append(normal_avg)

edge_features = np.stack(edge_features, axis=0)  # shape: (num_edges, 2)

# Prepare node features.
# Here we concatenate the node positions with the computed nodal area.
# If points have more than 2 dimensions, adjust accordingly.
node_features = np.concatenate([points[:, :2], node_area[:, None]], axis=1)

# Convert everything to torch tensors.
node_features = torch.tensor(node_features, dtype=torch.float)
edge_index = torch.tensor(edge_list, dtype=torch.long)
edge_attr = torch.tensor(edge_features, dtype=torch.float)

# Create the torch_geometric Data object.
data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

# The 'data' object now contains:
# - data.x: [num_nodes, 3] (2D position and area)
# - data.edge_index: [2, num_edges]
# - data.edge_attr: [num_edges, 2] (edge (face) normal direction)

print(data)
