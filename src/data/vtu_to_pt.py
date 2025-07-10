import os
import re
import argparse
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
import torch
import numpy as np
import pyvista as pv
from torch_geometric.data import Data
from tqdm import tqdm
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

# Suppress PyVista warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="pyvista")

# List of (Re, AoA) pairs to exclude (non-convergent or unphysical cases)
excluded_cases = [
    # (300000, 14), (150000, 14), (200000, 13),
    # (250000, 12), (550000, 8), (600000, 14),
    # (750000, 3), (800000, 6), (900000, 0),
    # (950000, 1), (100000, 12), (100000, 14),
]

def compute_implicit_distance(fluid_coords, surface_coords):
    from scipy.spatial import cKDTree
    tree = cKDTree(surface_coords)
    dists, _ = tree.query(fluid_coords)
    return dists

def log_scaling_distance(dist):
    return np.log1p(dist)

def scale_dataset(data, scaler=None, method='robust', return_scaler=False):
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a NumPy array")

    N, V, F = data.shape
    data_flat = data.reshape(-1, F)

    if scaler is None:
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        data_scaled_flat = scaler.fit_transform(data_flat)
    else:
        data_scaled_flat = scaler.transform(data_flat)

    data_scaled = data_scaled_flat.reshape(N, V, F)

    if return_scaler:
        return data_scaled, scaler
    else:
        return data_scaled

def find_vtu_files_single_folder(base_directory, max_aoa=15.0):
    vtu_files = []
    filtered_count = 0

    files = os.listdir(base_directory)
    print(f"Found {len(files)} files in directory")
    print(f"First few files: {files[:5]}")
    print(f"Filtering for AoA <= {max_aoa}°")

    for file in files:
        if file.lower().endswith('.vtu') and file.startswith('flow_Re_'):
            file_path = os.path.join(base_directory, file)

            base_name = os.path.splitext(file)[0]
            pattern = r'flow_Re_(\d+(?:\.\d+)?)_alpha_(-?\d+(?:\.\d+)?)'
            match = re.search(pattern, base_name)

            if match:
                reynolds = float(match.group(1))
                alpha = float(match.group(2))
                re_int = int(round(reynolds))
                alpha_int = int(round(alpha))

                if (re_int, alpha_int) in [(int(re), int(aoa)) for (re, aoa) in excluded_cases]:
                    print(f"❌ Excluded manually: {file} -> Re={re_int}, alpha={alpha_int}°")
                    filtered_count += 1
                    continue

                if abs(alpha) > max_aoa:
                    print(f"  ⏭️ Skipping {file} -> Re={reynolds:.1e}, alpha={alpha:.1f}° (exceeds max AoA of {max_aoa}°)")
                    filtered_count += 1
                    continue

                vtu_files.append((file_path, reynolds, alpha))
                print(f"  ✅ Found: {file} -> Re={reynolds:.1e}, alpha={alpha:.1f}°")
            else:
                print(f"  ⚠️ Could not extract flow parameters from filename: {file}")

    print(f"\n📊 File Filtering Summary:")
    print(f"  Total VTU files found: {len(files)}")
    print(f"  Files with AoA <= {max_aoa}°: {len(vtu_files)}")
    print(f"  Files filtered out (AoA > {max_aoa}° or manually excluded): {filtered_count}")

    return vtu_files

def build_graph_data(ux, uy, cp, coords, edge_index, re, alpha, variable):
    """Build graph data for a single VTU file."""
    # Compute distance to surface (where velocity is zero)
    surface_mask = (ux.flatten() == 0) & (uy.flatten() == 0)
    if np.any(surface_mask):
        surface_coords = coords[surface_mask]
        dist = compute_implicit_distance(coords, surface_coords)
    else:
        # Fallback: use distance to origin
        dist = np.linalg.norm(coords, axis=1)
    
    dist_log = log_scaling_distance(dist).reshape(-1, 1)

    re_feature = np.log10(re) * np.ones((coords.shape[0], 1))
    alpha_feature = np.deg2rad(alpha) * np.ones((coords.shape[0], 1))
    coords_x = coords[:, 0].reshape(-1, 1)
    coords_y = coords[:, 1].reshape(-1, 1)

    if variable == "re_x":
        x = np.concatenate([coords_x, coords_y, dist_log, re_feature, alpha_feature], axis=1)
        y = ux
    elif variable == "re_y":
        x = np.concatenate([coords_x, coords_y, dist_log, re_feature, alpha_feature], axis=1)
        y = uy
    elif variable == "re_p":
        x = np.concatenate([coords_x, coords_y, dist_log, re_feature, alpha_feature], axis=1)
        y = cp
    else:
        raise ValueError(f"Unknown variable: {variable}")

    if edge_index is not None:
        edge_attr = np.abs(coords[edge_index[1]] - coords[edge_index[0]])
        edge_weight = np.linalg.norm(edge_attr, axis=1)
    else:
        edge_attr = np.zeros((0, 2))
        edge_weight = np.zeros(0)

    return Data(
        x=torch.tensor(x, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.float32),
        pos=torch.tensor(coords, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long) if edge_index is not None else torch.zeros((2, 0), dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        edge_weight=torch.tensor(edge_weight, dtype=torch.float32),
        params=torch.tensor([re, alpha], dtype=torch.float32),
        surface_mask=torch.tensor(surface_mask, dtype=torch.bool)
    )

def vtu_to_pt(vtu_file_directory, 
              output_dir, 
              variable='re_x',
              vtu_array_name='Velocity',
              scaling_method='robust',
              train_ratio=0.9,
              overwrite=False,
              max_aoa=15.0):
    """
    Convert VTU files to PyTorch Geometric format with scaling and train/val split.
    
    Args:
        vtu_file_directory (str): Directory containing VTU files with format flow_Re_XX_alpha_YY.vtu
        output_dir (str): Directory to save train.pt, val.pt, and scaler.pkl
        variable (str): Target variable ('re_x', 're_y', 're_p')
        vtu_array_name (str): Name of the velocity array in VTU files
        scaling_method (str): Scaling method: 'standard', 'minmax', or 'robust'
        train_ratio (float): Ratio of data to use for training (0.0 to 1.0)
        overwrite (bool): Whether to overwrite existing files
        max_aoa (float): Maximum angle of attack to include (default: 15.0 degrees)
        
    Returns:
        tuple: (train_pt_path, val_pt_path, scaler_path) or (None, None, None) if failed
    """
    train_pt_path = os.path.join(output_dir, 'train.pt')
    val_pt_path = os.path.join(output_dir, 'val.pt')
    scaler_path = os.path.join(output_dir, 'scaler.pkl')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if not overwrite:
        if os.path.exists(train_pt_path) or os.path.exists(val_pt_path):
            print(f"PT files in {output_dir} already exist. Set overwrite=True to overwrite.")
            return None, None, None

    # Find all VTU files in single directory
    print("Pass 1: Finding all VTU files in directory...")
    print(f"Searching in directory: {vtu_file_directory}")
    vtu_files_info = find_vtu_files_single_folder(vtu_file_directory, max_aoa=max_aoa)
    
    if not vtu_files_info:
        print(f"No VTU files found in {vtu_file_directory}")
        return None, None, None

    all_graph_data = []  # List to store all graph data
    flow_params_map = {}  # Dict: {traj_num: (reynolds, alpha)}
    trajectory_numbers = []  # List of integer trajectory numbers

    print("Pass 2: Reading all VTU files and extracting data...")
    for file_path, reynolds, alpha in tqdm(vtu_files_info, desc="Reading VTUs"):
        try:
            traj_num = len(trajectory_numbers) + 1  # Use sequential numbering
            
            grid = pv.UnstructuredGrid(file_path)
            grid = grid.clip_box(bounds=(-0.5, 2, -1, 1, 0, 0), invert=False, crinkle=True)
            if grid is None or grid.points is None or vtu_array_name not in grid.point_data:
                 print(f"Warning: Failed to read {file_path} correctly or missing data. Skipping.")
                 continue

            # Extract coordinates and data
            coords = np.array(grid.points[:, 0:2], dtype=np.float32)
            
            # Extract velocity and pressure data
            raw_velocities = np.array(grid.point_data[vtu_array_name], dtype=np.float32)
            raw_pressure = np.array(grid.point_data['Pressure_Coefficient'], dtype=np.float32)
            
            ux = raw_velocities[:, 0].reshape(-1, 1)
            uy = raw_velocities[:, 1].reshape(-1, 1)
            cp = raw_pressure.reshape(-1, 1)
            
            # Extract edge connectivity
            edges = grid.extract_all_edges()
            if edges is not None and hasattr(edges, 'lines') and edges.lines is not None:
                edge_points = edges.lines.reshape(-1, 3)[1:]  # Skip first element which is line count
                edge_index = edge_points[:, 1:].reshape(2, -1)
            else:
                print(f"Warning: Could not extract edge connectivity from {file_path}")
                edge_index = None
            
            # Build graph data
            graph_data = build_graph_data(ux, uy, cp, coords, edge_index, reynolds, alpha, variable)
            
            trajectory_numbers.append(traj_num)
            flow_params_map[traj_num] = (reynolds, alpha)
            all_graph_data.append(graph_data)
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}. Skipping.")
            continue
            
    if not trajectory_numbers:
        print("No valid trajectory data could be extracted after Pass 2.")
        return None, None, None
    
    # Print summary of flow parameters
    print(f"\n📊 Flow Parameters Summary:")
    unique_params = set(flow_params_map.values())
    for reynolds, alpha in sorted(unique_params):
        count = sum(1 for params in flow_params_map.values() if params == (reynolds, alpha))
        print(f"  Re={reynolds:.1e}, alpha={alpha:.1f}°: {count} files")
    
    # Prepare data for scaling
    print("Pass 3: Preparing data for scaling...")
    
    # Extract features for scaling (x features)
    all_features = []
    all_targets = []
    
    for graph in all_graph_data:
        features = graph.x.numpy()  # [num_nodes, num_features]
        targets = graph.y.numpy()   # [num_nodes, 1]
        
        all_features.append(features)
        all_targets.append(targets)
    
    # Stack all features and targets
    stacked_features = np.concatenate(all_features, axis=0)  # [total_nodes, num_features]
    stacked_targets = np.concatenate(all_targets, axis=0)    # [total_nodes, 1]
    
    print(f"Scaling features of shape: {stacked_features.shape}")
    print(f"Scaling targets of shape: {stacked_targets.shape}")
    
    # Scale features and targets
    scaled_features, feature_scaler = scale_dataset(
        stacked_features.reshape(1, -1, stacked_features.shape[1]), 
        scaler=None, 
        method=scaling_method, 
        return_scaler=True
    )
    
    scaled_targets, target_scaler = scale_dataset(
        stacked_targets.reshape(1, -1, stacked_targets.shape[1]), 
        scaler=None, 
        method=scaling_method, 
        return_scaler=True
    )
    
    # Reshape back
    scaled_features = scaled_features.reshape(-1, stacked_features.shape[1])
    scaled_targets = scaled_targets.reshape(-1, stacked_targets.shape[1])
    
    # Update graph data with scaled features and targets
    print("Pass 4: Updating graphs with scaled data...")
    start_idx = 0
    for i, graph in enumerate(all_graph_data):
        num_nodes = graph.x.shape[0]
        end_idx = start_idx + num_nodes
        
        # Update features and targets
        graph.x = torch.tensor(scaled_features[start_idx:end_idx], dtype=torch.float32)
        graph.y = torch.tensor(scaled_targets[start_idx:end_idx], dtype=torch.float32)
        
        start_idx = end_idx
    
    # Save scalers
    scalers = {
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler
    }
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scalers, f)
    print(f"Scalers saved to {scaler_path}")
    
    # Split data into train/val
    print("Pass 5: Splitting data into train/val...")
    from collections import defaultdict
    
    # Group by flow parameters for uniform splitting
    param_groups = defaultdict(list)
    for traj_id in trajectory_numbers:
        param = flow_params_map[traj_id]
        param_groups[param].append(traj_id)
    
    train_indices = []
    val_indices = []

    for group in param_groups.values():
        group = list(group)
        np.random.shuffle(group)

        # Ensure at least 1 trajectory per group goes to train if possible
        if len(group) == 1:
            # Assign to train with some probability based on train_ratio
            if np.random.rand() < train_ratio:
                train_indices.append(group[0])
            else:
                val_indices.append(group[0])
        else:
            train_size = max(1, int(len(group) * train_ratio))
            train_indices.extend(group[:train_size])
            val_indices.extend(group[train_size:])

    # Shuffle final indices
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    
    # Create train and val data lists
    train_data = [all_graph_data[i-1] for i in train_indices]  # i-1 because trajectory numbers start from 1
    val_data = [all_graph_data[i-1] for i in val_indices]
    
    # Save train and val data
    print(f"Writing training data to {train_pt_path} ({len(train_data)} graphs)")
    torch.save(train_data, train_pt_path)
    
    print(f"Writing validation data to {val_pt_path} ({len(val_data)} graphs)")
    torch.save(val_data, val_pt_path)
    
    print(f"VTU to PT conversion with scaling and splitting complete. Output in {output_dir}")
    return train_pt_path, val_pt_path, scaler_path

def main():
    """Main function to run the VTU to PT conversion."""
    parser = argparse.ArgumentParser(description="Convert VTU files to PyTorch Geometric format with scaling")
    parser.add_argument("vtu_dir", help="Directory containing VTU files")
    parser.add_argument("output_dir", help="Output directory for PT files")
    parser.add_argument("--variable", default="re_x", choices=["re_x", "re_y", "re_p"], 
                       help="Target variable (default: re_x)")
    parser.add_argument("--scaling", default="robust", choices=["standard", "minmax", "robust"],
                       help="Scaling method (default: robust)")
    parser.add_argument("--train-ratio", type=float, default=0.9,
                       help="Training data ratio (default: 0.9)")
    parser.add_argument("--max-aoa", type=float, default=15.0,
                       help="Maximum angle of attack to include (default: 15.0)")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing output files")
    
    args = parser.parse_args()
    
    try:
        train_path, val_path, scaler_path = vtu_to_pt(
            vtu_file_directory=args.vtu_dir,
            output_dir=args.output_dir,
            variable=args.variable,
            scaling_method=args.scaling,
            train_ratio=args.train_ratio,
            overwrite=args.overwrite,
            max_aoa=args.max_aoa
        )
        
        if train_path and val_path and scaler_path:
            print(f"\n✅ Conversion successful!")
            print(f"Train data: {train_path}")
            print(f"Val data: {val_path}")
            print(f"Scalers: {scaler_path}")
        else:
            print(f"\n❌ Conversion failed!")
            
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())