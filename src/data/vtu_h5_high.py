import h5py
import pyvista as pv
import os
import tqdm
import numpy as np
import torch
import pickle
import re
from src.data import scaling

def split_dataset_trajectories(all_trajectory_indices, train_ratio):
    """
    Split trajectory indices into training and validation sets.
    
    Args:
        all_trajectory_indices: List of trajectory indices to split
        train_ratio: Ratio of data to use for training (0.0 to 1.0)
    
    Returns:
        tuple: (train_indices, val_indices) - sorted lists of trajectory numbers
    """
    total_sims = len(all_trajectory_indices)
    train_sims = int(train_ratio * total_sims)
    
    # Create a mutable copy and shuffle
    shuffled_indices = list(all_trajectory_indices)
    np.random.shuffle(shuffled_indices)
    
    # Split and sort for consistent ordering
    train_indices = sorted(shuffled_indices[:train_sims])
    val_indices = sorted(shuffled_indices[train_sims:])
    
    return train_indices, val_indices

def write_single_h5_file(output_h5_path, flow_params_to_write, 
                        all_scaled_data_map, 
                        all_coordinates_map, 
                        edge_index):
    """
    Write trajectory data to a single H5 file (train or validation).
    
    Args:
        output_h5_path: Path to the output H5 file
        flow_params_to_write: List of (reynolds, alpha) tuples to write
        all_scaled_data_map: Dict mapping flow parameters to scaled data dict
        all_coordinates_map: Dict mapping flow parameters to coordinate arrays
        edge_index: Edge connectivity information
    """
    with h5py.File(output_h5_path, 'w') as f:
        for flow_params in tqdm.tqdm(flow_params_to_write, 
                                    desc=f"Writing {os.path.basename(output_h5_path)}"):
            reynolds, alpha = flow_params
            # Create group with Re_XX_alpha_YY format
            group_name = f'Re_{reynolds}_alpha_{alpha}'
            g = f.create_group(group_name)
            
            # Write coordinates and edge information
            g['coordinates'] = all_coordinates_map[flow_params]
            if edge_index is not None:
                g['edge_index'] = edge_index
            
            # Write flow parameters
            g['parameters'] = np.array([reynolds, alpha])
            
            # Write individual variables
            scaled_data = all_scaled_data_map[flow_params]
            for var_name, var_data in scaled_data.items():
                if isinstance(var_data, torch.Tensor):
                    g[var_name] = var_data.numpy()
                else:
                    g[var_name] = var_data

def extract_flow_parameters_from_path(file_path):
    """
    Extract Reynolds number and angle of attack from ReXX/ReXX_alpha_YY/flow.vtu path structure.
    
    Args:
        file_path: Path to VTU file in format ReXX/ReXX_alpha_YY/flow.vtu
    
    Returns:
        tuple: (reynolds_number, angle_of_attack) or (None, None) if parsing fails
    """
    # Split path into components
    path_parts = file_path.replace('\\', '/').split('/')
    
    # Look for the directory pattern ReXX_alpha_YY
    for part in path_parts:
        # Pattern to match ReXX_alpha_YY
        pattern = r'Re(\d+(?:\.\d+)?)_alpha_(-?\d+(?:\.\d+)?)'
        match = re.match(pattern, part)
        
        if match:
            reynolds = float(match.group(1))
            alpha = float(match.group(2))
            return reynolds, alpha
    
    return None, None

def find_vtu_files_recursive(directory):
    """
    Recursively find all VTU files in the directory structure.
    
    Args:
        directory: Root directory to search
    
    Returns:
        list: List of full paths to VTU files
    """
    vtu_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.vtu'):
                full_path = os.path.join(root, file)
                vtu_files.append(full_path)
    
    return sorted(vtu_files)



def vtu_to_h5(vtu_file_directory, 
              output_h5_dir,
              vtu_array_name='Velocity',
              train_ratio=0.9, 
              scaling_type=4, 
              scaler_name='standard',
              overwrite=False):
    """
    Convert VTU files to H5 format with train/validation split and scaling.
    
    Args:
        vtu_file_directory: Directory containing VTU files in structure ReXX/ReXX_alpha_YY/flow.vtu
        output_h5_dir: Directory to save train.h5, val.h5, and scaler.pkl
        vtu_array_name: Name of the velocity array in VTU files
        train_ratio: Ratio of data to use for training (0.0 to 1.0)
        scaling_type: Type of scaling to apply
        scaler_name: Name of the scaler to use
        overwrite: Whether to overwrite existing files
    
    Returns:
        tuple: (train_h5_path, val_h5_path, scaler_path) or (None, None, None) on failure
    """
    # Setup output paths
    train_h5_path = os.path.join(output_h5_dir, 'train.h5')
    val_h5_path = os.path.join(output_h5_dir, 'val.h5')
    scaler_path = os.path.join(output_h5_dir, 'scaler.pkl')

    # Create output directory if it doesn't exist
    if not os.path.exists(output_h5_dir):
        os.makedirs(output_h5_dir)
    
    # Check if files already exist
    if not overwrite:
        if os.path.exists(train_h5_path) or os.path.exists(val_h5_path):
            print(f"H5 files in {output_h5_dir} already exist. Set overwrite=True to overwrite.")
            return None, None, None

    # Get list of VTU files recursively
    vtu_files = find_vtu_files_recursive(vtu_file_directory)
    if not vtu_files:
        print(f"No VTU files found in {vtu_file_directory}")
        return None, None, None

    print(f"Found {len(vtu_files)} VTU files in directory structure")

    # Data storage
    all_trajectory_data_raw = {}  # {flow_params: {'coordinates': ndarray, 'variables': dict}}
    flow_parameters_list = []     # List of (reynolds, alpha) tuples
    edge_index = None

    print("Pass 1: Reading all VTU files and extracting data...")
    for vtu_path in tqdm.tqdm(vtu_files, desc="Reading VTUs"):
        try:
            # Extract flow parameters from path
            reynolds, alpha = extract_flow_parameters_from_path(vtu_path)
            if reynolds is None or alpha is None:
                print(f"Warning: Could not parse flow parameters from {vtu_path}. Skipping.")
                continue

            flow_params = (reynolds, alpha)
            
            # Read VTU file
            grid = pv.UnstructuredGrid(vtu_path)
            
            # Validate grid data
            if grid is None or grid.points is None or vtu_array_name not in grid.point_data:
                print(f"Warning: Failed to read {vtu_path} correctly or missing data. Skipping.")
                continue

            # Extract coordinates (2D)
            coordinates = np.array(grid.points[:, 0:2], dtype=np.float32) 
            
            # Extract edge connectivity (only once, should be same for all files)
            if edge_index is None:
                edges = grid.extract_all_edges()
                if edges is not None and hasattr(edges, 'lines') and edges.lines is not None:
                    edge_points = edges.lines.reshape(-1, 3)[1:]  # Skip first element (line count)
                    edge_index = edge_points[:, 1:].reshape(2, -1)
                else:
                    print(f"Warning: Could not extract edge connectivity from {vtu_path}")
                    edge_index = None
            
            # Extract velocity and pressure data
            raw_velocities = np.array(grid.point_data[vtu_array_name], dtype=np.float32)
            surface_mask = raw_velocities[:, 0] == 0.0
            
            # Extract individual variables
            ux = raw_velocities[:, 0]
            uy = raw_velocities[:, 1]
            
            # Extract pressure and coefficients if available
            variables = {
                'Ux': torch.tensor(ux, dtype=torch.float32),
                'Uy': torch.tensor(uy, dtype=torch.float32)
            }
            
            # Try to extract additional variables if they exist
            if 'Pressure' in grid.point_data:
                raw_pressure = np.array(grid.point_data['Pressure'], dtype=np.float32)
                variables['Pressure'] = torch.tensor(raw_pressure, dtype=torch.float32)
            
            if 'Pressure_Coefficient' in grid.point_data:
                cp = np.array(grid['Pressure_Coefficient'], dtype=np.float32)
                cp[~surface_mask] = 0.0  # Set non-surface points to 0
                variables['Cp'] = torch.tensor(cp, dtype=torch.float32)
            
            if 'Skin_Friction_Coefficient' in grid.point_data:
                cf = np.array(grid['Skin_Friction_Coefficient'], dtype=np.float32)[:, :2]
                cf[~surface_mask] = 0.0  # Set non-surface points to 0
                variables['Cf'] = torch.tensor(cf, dtype=torch.float32)
            
            # Check for consistent number of nodes across all files
            if flow_parameters_list and coordinates.shape[0] != all_trajectory_data_raw[flow_parameters_list[0]]['coordinates'].shape[0]:
                print(f"Error: Inconsistent number of nodes in {vtu_path}. "
                      f"Expected {all_trajectory_data_raw[flow_parameters_list[0]]['coordinates'].shape[0]}, "
                      f"got {coordinates.shape[0]}. Skipping file.")
                continue

            # Store data
            flow_parameters_list.append(flow_params)
            all_trajectory_data_raw[flow_params] = {
                'coordinates': coordinates,
                'edge_index': edge_index,
                'variables': variables
            }
            
        except Exception as e:
            print(f"Error processing file {vtu_path}: {e}. Skipping.")
            continue
            
    if not flow_parameters_list:
        print("No valid trajectory data could be extracted after Pass 1.")
        return None, None, None
    
    # Sort flow parameters for consistent ordering
    flow_parameters_list.sort()

    # Get list of all variables
    first_flow_params = flow_parameters_list[0]
    variable_names = list(all_trajectory_data_raw[first_flow_params]['variables'].keys())
    print(f"Found variables: {variable_names}")

    # Split flow parameters for train/validation sets
    train_flow_params, val_flow_params = split_dataset_trajectories(flow_parameters_list, train_ratio)

    # Initialize scaling variables
    scalers = {}  # Dict to store scalers for each variable
    all_scaled_data = {}  # Dict to store scaled data for each flow parameter

    # Perform scaling if training data exists
    if not train_flow_params:
        print("Warning: No training trajectories after splitting. Scaling will be skipped.")
        # Use unscaled data
        for fp in flow_parameters_list:
            all_scaled_data[fp] = all_trajectory_data_raw[fp]['variables']
    else:
        print("Fitting scalers for each variable...")
        
        # Fit scalers for each variable separately
        for var_name in variable_names:
            print(f"Fitting scaler for {var_name}...")
            
            # Stack data for this variable across all training trajectories
            train_var_data = []
            for fp in train_flow_params:
                var_data = all_trajectory_data_raw[fp]['variables'][var_name]
                if var_data.dim() == 1:
                    var_data = var_data.unsqueeze(1)  # Add feature dimension
                train_var_data.append(var_data)
            
            # Stack and reshape for scaling
            stacked_var_data = torch.stack(train_var_data, dim=0)  # [num_train_traj, num_nodes, num_features]
            reshaped_var_data = stacked_var_data.reshape(-1, stacked_var_data.shape[-1])
            
            print(f"  Training data shape for {var_name}: {reshaped_var_data.shape}")
            
            # Fit scaler
            scaler, _ = scaling.tensor_scaling(reshaped_var_data, scaling_type, scaler_name)
            scalers[var_name] = scaler
            
            if scaler:
                print(f"  Scaler fitted for {var_name}")
            else:
                print(f"  Warning: Scaler fitting failed for {var_name}")
        
        # For now, skip scaling to get basic functionality working
        print("Skipping scaling for now - using unscaled data...")
        for fp in flow_parameters_list:
            all_scaled_data[fp] = all_trajectory_data_raw[fp]['variables']

    # Save the fitted scalers (even though we're not using them for now)
    if scalers:
        with open(scaler_path, 'wb') as f_scaler:
            pickle.dump(scalers, f_scaler)
        print(f"Scalers saved to {scaler_path} (not currently used)")
    else:
        print("No scalers to save")

    # Prepare data maps for writing
    all_coordinates_map = {
        fp: all_trajectory_data_raw[fp]['coordinates'] 
        for fp in flow_parameters_list
    }

    # Write training data
    print(f"Writing training data to {train_h5_path} ({len(train_flow_params)} trajectories)")
    write_single_h5_file(train_h5_path, train_flow_params, 
                        all_scaled_data, all_coordinates_map, edge_index)
    
    # Write validation data
    print(f"Writing validation data to {val_h5_path} ({len(val_flow_params)} trajectories)")
    write_single_h5_file(val_h5_path, val_flow_params, 
                        all_scaled_data, all_coordinates_map, edge_index)

    print(f"VTU to H5 conversion with scaling and splitting complete. Output in {output_h5_dir}")
    return train_h5_path, val_h5_path, scaler_path
