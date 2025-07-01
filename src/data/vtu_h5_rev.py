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
                        all_scaled_velocities_map, 
                        all_coordinates_map, 
                        edge_index):
    """
    Write trajectory data to a single H5 file (train or validation).
    
    Args:
        output_h5_path: Path to the output H5 file
        flow_params_to_write: List of (reynolds, alpha) tuples to write
        all_scaled_velocities_map: Dict mapping flow parameters to scaled velocity tensors
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
            
            # Extract velocity components from the scaled tensor
            velocities_for_traj = all_scaled_velocities_map[flow_params]  # [num_nodes, num_components]
            
            # Write individual velocity components
            g['Ux'] = velocities_for_traj[:, 0].numpy()
            g['Uy'] = velocities_for_traj[:, 1].numpy()
            g['Pressure'] = velocities_for_traj[:, 2].numpy()
            g['Cp'] = velocities_for_traj[:, 3].numpy()
            g['Cf'] = velocities_for_traj[:, 4:6].numpy()  # Skin friction coefficients

def extract_flow_parameters_from_filename(filename):
    """
    Extract Reynolds number and angle of attack from flow_Re_XX_alpha_YY.vtu filename.
    
    Args:
        filename: VTU filename in format flow_Re_XX_alpha_YY.vtu
    
    Returns:
        tuple: (reynolds_number, angle_of_attack) or (None, None) if parsing fails
    """
    # Pattern to match flow_Re_XX_alpha_YY.vtu
    pattern = r'flow_Re_(\d+(?:\.\d+)?)_alpha_(-?\d+(?:\.\d+)?)\.vtu'
    match = re.match(pattern, filename)
    
    if match:
        reynolds = float(match.group(1))
        alpha = float(match.group(2))
        return reynolds, alpha
    else:
        return None, None

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
        vtu_file_directory: Directory containing VTU files
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

    # Get list of VTU files
    vtu_files = sorted([f for f in os.listdir(vtu_file_directory) 
                       if f.lower().endswith('.vtu')])
    if not vtu_files:
        print(f"No VTU files found in {vtu_file_directory}")
        return None, None, None

    # Data storage
    all_trajectory_data_raw = {}  # {flow_params: {'coordinates': ndarray, 'velocities': tensor}}
    flow_parameters_list = []     # List of (reynolds, alpha) tuples
    edge_index = None

    print("Pass 1: Reading all VTU files and extracting data...")
    for vtu_file in tqdm.tqdm(vtu_files, desc="Reading VTUs"):
        try:
            # Extract flow parameters from filename
            reynolds, alpha = extract_flow_parameters_from_filename(vtu_file)
            if reynolds is None or alpha is None:
                print(f"Warning: Could not parse flow parameters from {vtu_file}. Skipping.")
                continue

            flow_params = (reynolds, alpha)
            
            # Read VTU file
            vtu_path = os.path.join(vtu_file_directory, vtu_file)
            grid = pv.UnstructuredGrid(vtu_path)
            
            # Validate grid data
            if grid is None or grid.points is None or vtu_array_name not in grid.point_data:
                print(f"Warning: Failed to read {vtu_file} correctly or missing data. Skipping.")
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
                    print(f"Warning: Could not extract edge connectivity from {vtu_file}")
                    edge_index = None
            
            # Extract velocity and pressure data
            raw_velocities = np.array(grid.point_data[vtu_array_name], dtype=np.float32)
            surface_mask = raw_velocities[:, 0] == 0.0
            
            # Extract pressure and coefficients
            raw_pressure = np.array(grid.point_data['Pressure'], dtype=np.float32)
            cp = np.array(grid['Pressure_Coefficient'], dtype=np.float32)
            cp[~surface_mask] = 0.0  # Set non-surface points to 0
            
            cf = np.array(grid['Skin_Friction_Coefficient'], dtype=np.float32)[:, :2]
            cf[~surface_mask] = 0.0  # Set non-surface points to 0

            # Combine all velocity components
            velocities_combined = np.concatenate([
                raw_velocities[:, :2],      # Ux, Uy
                raw_pressure[:, None],      # Pressure
                cp[:, None],                # Cp
                cf[:, :2]                   # Cf (2 components)
            ], axis=1)
            
            # Check for consistent number of nodes across all files
            if flow_parameters_list and coordinates.shape[0] != all_trajectory_data_raw[flow_parameters_list[0]]['coordinates'].shape[0]:
                print(f"Error: Inconsistent number of nodes in {vtu_file}. "
                      f"Expected {all_trajectory_data_raw[flow_parameters_list[0]]['coordinates'].shape[0]}, "
                      f"got {coordinates.shape[0]}. Skipping file.")
                continue

            # Store data
            flow_parameters_list.append(flow_params)
            all_trajectory_data_raw[flow_params] = {
                'coordinates': coordinates,
                'edge_index': edge_index,
                'velocities': torch.tensor(velocities_combined, dtype=torch.float32)
            }
            
        except Exception as e:
            print(f"Error processing file {vtu_file}: {e}. Skipping.")
            continue
            
    if not flow_parameters_list:
        print("No valid trajectory data could be extracted after Pass 1.")
        return None, None, None
    
    # Sort flow parameters for consistent ordering
    flow_parameters_list.sort()

    # Stack all velocities for scaling
    stacked_velocities_list = [all_trajectory_data_raw[fp]['velocities'] 
                              for fp in flow_parameters_list]
    
    # Validate shapes before stacking
    first_tensor_shape = stacked_velocities_list[0].shape
    if not all(t.shape == first_tensor_shape for t in stacked_velocities_list):
        print("Error: Velocity tensors have inconsistent shapes across trajectories. Cannot stack for scaling.")
        return None, None, None
    
    num_velocity_components = first_tensor_shape[1]
    velocities_to_scale = torch.stack(stacked_velocities_list, dim=0)

    # Split flow parameters for train/validation sets
    train_flow_params, val_flow_params = split_dataset_trajectories(flow_parameters_list, train_ratio)

    # Initialize scaling variables
    scaler = None
    scaled_all_velocities_tensor = velocities_to_scale  # Default if no scaling

    # Perform scaling if training data exists
    if not train_flow_params:
        print("Warning: No training trajectories after splitting. Scaling will be skipped.")
    else:
        # Get training data indices
        train_indices_in_stacked_tensor = [flow_parameters_list.index(fp) 
                                          for fp in train_flow_params]
        train_velocities_for_scaler = velocities_to_scale[train_indices_in_stacked_tensor]
        
        # Reshape for scaler fitting
        original_shape_train = train_velocities_for_scaler.shape
        reshaped_train_velocities = train_velocities_for_scaler.reshape(-1, num_velocity_components)
        
        print(f"Fitting scaler on training data of shape: {reshaped_train_velocities.shape}")
        scaler, _ = scaling.tensor_scaling(reshaped_train_velocities, scaling_type, scaler_name)
        
        if scaler:
            print("Scaler fitted. Applying to the entire dataset.")
            original_shape_all = velocities_to_scale.shape
            reshaped_all_velocities = velocities_to_scale.reshape(-1, num_velocity_components)
            
            # Apply scaling transformation
            # Handle both single scaler and list of scalers
            if isinstance(scaler, list):
                # For scaling_type 3 and 4, we have a list of scalers
                # Apply the scalers in the correct order
                if scaling_type == 3:  # FEATURE-SAMPLE SCALING
                    temp = scaler[0].transform(reshaped_all_velocities.numpy())
                    scaled_transformed_data = scaler[1].transform(temp)
                elif scaling_type == 4:  # SAMPLE-FEATURE SCALING
                    temp = scaler[0].transform(reshaped_all_velocities.numpy())
                    temp_np = np.array(temp)
                    temp_transposed = np.transpose(temp_np)
                    scaled_transformed_data = np.transpose(scaler[1].transform(temp_transposed))
                else:
                    print(f"Warning: Unknown scaling type {scaling_type} with list of scalers")
                    scaled_transformed_data = reshaped_all_velocities.numpy()
            elif hasattr(scaler, 'transform'):
                # Single scaler object
                scaled_transformed_data = scaler.transform(reshaped_all_velocities.numpy())
            else:
                print("Warning: Fitted scaler does not have a 'transform' method. "
                      "Scaling might not be applied correctly. Check 'src.data.scaling' module.")
                scaled_transformed_data = reshaped_all_velocities.numpy()
            
            scaled_all_velocities_tensor = torch.tensor(scaled_transformed_data, 
                                                      dtype=torch.float32).reshape(original_shape_all)
        else:
            print("Warning: Scaler fitting failed. Scaling will be skipped.")

    # Save the fitted scaler
    if scaler:
        with open(scaler_path, 'wb') as f_scaler:
            pickle.dump(scaler, f_scaler)
        print(f"Scaler saved to {scaler_path}")

    # Prepare data maps for writing
    all_scaled_velocities_map = {
        fp: scaled_all_velocities_tensor[flow_parameters_list.index(fp)] 
        for fp in flow_parameters_list
    }
    all_coordinates_map = {
        fp: all_trajectory_data_raw[fp]['coordinates'] 
        for fp in flow_parameters_list
    }

    # Write training data
    print(f"Writing training data to {train_h5_path} ({len(train_flow_params)} trajectories)")
    write_single_h5_file(train_h5_path, train_flow_params, 
                        all_scaled_velocities_map, all_coordinates_map, edge_index)
    
    # Write validation data
    print(f"Writing validation data to {val_h5_path} ({len(val_flow_params)} trajectories)")
    write_single_h5_file(val_h5_path, val_flow_params, 
                        all_scaled_velocities_map, all_coordinates_map, edge_index)

    print(f"VTU to H5 conversion with scaling and splitting complete. Output in {output_h5_dir}")
    return train_h5_path, val_h5_path, scaler_path
