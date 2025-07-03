import h5py
import pyvista as pv
import os
import tqdm
import numpy as np
import torch
import pickle
import re
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def scale_dataset(data, scaler=None, method='standard', return_scaler=False):
    """
    Scale a dataset of shape [num_samples, num_nodes, num_features].

    Parameters:
        data (np.ndarray): Input data array of shape [N, V, F]
        scaler (sklearn Scaler or None): If provided, use this scaler to transform data
        method (str): Scaling method: 'standard', 'minmax', or 'robust'
        return_scaler (bool): If True, return the fitted scaler

    Returns:
        scaled_data (np.ndarray): Scaled data with same shape as input
        scaler (optional): Fitted scaler object
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a NumPy array")

    N, V, F = data.shape
    data_flat = data.reshape(-1, F)

    # Fit new scaler if not provided
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

# Helper function to split trajectory indices
def split_dataset_trajectories(all_trajectory_indices, train_ratio):
    total_sims = len(all_trajectory_indices)
    train_sims = int(train_ratio * total_sims)
    
    shuffled_indices = list(all_trajectory_indices) # Make a mutable copy
    np.random.shuffle(shuffled_indices)
    
    train_indices = sorted(shuffled_indices[:train_sims])
    val_indices = sorted(shuffled_indices[train_sims:])
    
    return train_indices, val_indices

# Helper function to write data to a single H5 file (train or val)
def write_single_h5_file(output_h5_path, trajectories_to_write, 
                         all_scaled_velocities_map, # Dict: {traj_num: scaled_vel_tensor [nodes, 2]}
                         all_coordinates_map,       # Dict: {traj_num: coord_array [nodes, 2]}
                         edge_index,
                         flow_params_map):          # Dict: {traj_num: (reynolds, alpha)}
    with h5py.File(output_h5_path, 'w') as f:
        # Track used group names to handle duplicates
        used_group_names = set()
        duplicate_count = 0
        
        for traj_num in tqdm.tqdm(trajectories_to_write, desc=f"Writing {os.path.basename(output_h5_path)}"):
            # Create group name in format Re_XX_alpha_YY
            reynolds, alpha = flow_params_map[traj_num]
            base_group_name = f'Re_{int(reynolds)}_alpha_{int(alpha)}'
            
            # Handle duplicate group names by adding a suffix
            group_name = base_group_name
            counter = 1
            while group_name in used_group_names:
                group_name = f'{base_group_name}_{counter}'
                counter += 1
            
            used_group_names.add(group_name)
            g = f.create_group(group_name)
            
            # Print warning if duplicate was detected
            if group_name != base_group_name:
                print(f"    ⚠️ Duplicate flow parameters detected. Using group name: {group_name}")
                duplicate_count += 1
            
            g['coordinates'] = all_coordinates_map[traj_num]
            if edge_index is not None:
                g['edge_index'] = edge_index
            
            # Write flow parameters
            g['parameters'] = np.array([reynolds, alpha])
            
            velocities_for_traj = all_scaled_velocities_map[traj_num] # Should be [num_nodes, 2]
            g['Ux'] = velocities_for_traj[:, 0].numpy() # Convert to numpy for h5py
            g['Uy'] = velocities_for_traj[:, 1].numpy()
            g['Pressure'] = velocities_for_traj[:, 2].numpy()
            g['Cp'] = velocities_for_traj[:, 3].numpy()
            g['Cf'] = velocities_for_traj[:, 4:6].numpy() # Assuming Cf has 2 components
            g['Y_Plus'] = velocities_for_traj[:, 6].numpy()
            g['Nu_Tilde'] = velocities_for_traj[:, 7].numpy()
            g['Eddy_Viscosity'] = velocities_for_traj[:, 8].numpy()
        
        # Print summary of duplicates
        if duplicate_count > 0:
            print(f"📊 Summary: {duplicate_count} duplicate flow parameter combinations were found and handled.")
        else:
            print(f"📊 Summary: No duplicate flow parameters found.")

def find_vtu_files_single_folder(base_directory):
    """
    Find all flow_Re_XX_alpha_YY.vtu files in a single directory
    
    Args:
        base_directory (str): Directory containing VTU files
        
    Returns:
        list: List of tuples (file_path, reynolds, alpha) for each flow_Re_XX_alpha_YY.vtu file found
    """
    vtu_files = []
    
    # Get all files in the directory
    files = os.listdir(base_directory)
    print(f"Found {len(files)} files in directory")
    print(f"First few files: {files[:5]}")
    
    for file in files:
        # Only process flow_Re_XX_alpha_YY.vtu files
        if file.lower().endswith('.vtu') and file.startswith('flow_Re_'):
            file_path = os.path.join(base_directory, file)
            
            # Extract Reynolds number and alpha from filename
            # Expected format: flow_Re_XX_alpha_YY.vtu
            base_name = os.path.splitext(file)[0]  # Remove .vtu extension
            
            # Pattern to match flow_Re_XX_alpha_YY
            pattern = r'flow_Re_(\d+(?:\.\d+)?)_alpha_(-?\d+(?:\.\d+)?)'
            match = re.search(pattern, base_name)
            
            if match:
                reynolds = float(match.group(1))
                alpha = float(match.group(2))
                vtu_files.append((file_path, reynolds, alpha))
                print(f"  ✅ Found: {file} -> Re={reynolds:.1e}, alpha={alpha:.1f}°")
            else:
                print(f"  ⚠️ Could not extract flow parameters from filename: {file}")
    
    return vtu_files

def vtu_to_h5(vtu_file_directory, 
              output_h5_dir, # Directory to save train.h5, val.h5, and scaler.pkl
              vtu_array_name='Velocity', # Name of the velocity array in VTU files
              train_ratio=0.9, 
              scaling_method='standard',
              overwrite=False):
    """
    Convert VTU files to H5 format with scaling and train/val split.
    
    Args:
        vtu_file_directory (str): Directory containing VTU files with format flow_Re_XX_alpha_YY.vtu
        output_h5_dir (str): Directory to save train.h5, val.h5, and scaler.pkl
        vtu_array_name (str): Name of the velocity array in VTU files
        train_ratio (float): Ratio of data to use for training (0.0 to 1.0)
        scaling_method (str): Scaling method: 'standard', 'minmax', or 'robust'
        overwrite (bool): Whether to overwrite existing files
        
    Returns:
        tuple: (train_h5_path, val_h5_path, scaler_path) or (None, None, None) if failed
    """
    edge_index = None
    train_h5_path = os.path.join(output_h5_dir, 'train.h5')
    val_h5_path = os.path.join(output_h5_dir, 'val.h5')
    scaler_path = os.path.join(output_h5_dir, 'scaler.pkl')

    if not os.path.exists(output_h5_dir):
        os.makedirs(output_h5_dir)
    
    if not overwrite:
        if os.path.exists(train_h5_path) or os.path.exists(val_h5_path):
            print(f"H5 files in {output_h5_dir} already exist. Set overwrite=True to overwrite.")
            return None, None, None

    # Find all VTU files in single directory
    print("Pass 1: Finding all VTU files in directory...")
    print(f"Searching in directory: {vtu_file_directory}")
    vtu_files_info = find_vtu_files_single_folder(vtu_file_directory)
    
    if not vtu_files_info:
        print(f"No VTU files found in {vtu_file_directory}")
        return None, None, None

    all_trajectory_data_raw = {} # Stores raw data: {traj_num: {'coordinates': ndarray, 'velocities': tensor}}
    trajectory_numbers = [] # List of integer trajectory numbers
    flow_params_map = {} # Dict: {traj_num: (reynolds, alpha)}

    print("Pass 2: Reading all VTU files and extracting data...")
    for file_path, reynolds, alpha in tqdm.tqdm(vtu_files_info, desc="Reading VTUs"):
        try:
            traj_num = len(trajectory_numbers) + 1  # Use sequential numbering
            
            grid = pv.UnstructuredGrid(file_path)
            if grid is None or grid.points is None or vtu_array_name not in grid.point_data:
                 print(f"Warning: Failed to read {file_path} correctly or missing data. Skipping.")
                 continue

            #coordinates
            coordinates = np.array(grid.points[:, 0:2], dtype=np.float32) 
            
            if edge_index is None:
                #edge index
                edges = grid.extract_all_edges()
                if edges is not None and hasattr(edges, 'lines') and edges.lines is not None:
                    edge_points = edges.lines.reshape(-1, 3)[1:] # Skip first element which is line count
                    edge_index = edge_points[:,1:].reshape(2, -1)
                else:
                    print(f"Warning: Could not extract edge connectivity from {file_path}")
                    edge_index = None
            #features: coord, ux, uy, Y_plus, nutilde, eddy viscosity
            #target: Cf, Cp
            #velocity
            raw_velocities = np.array(grid.point_data[vtu_array_name], dtype=np.float32)
            raw_pressure = np.array(grid.point_data['Pressure'], dtype=np.float32)
            cp = np.array(grid['Pressure_Coefficient'], dtype=np.float32)
            cf = np.array(grid['Skin_Friction_Coefficient'], dtype=np.float32)[:,:2]
            Y_plus = np.array(grid['Y_Plus'], dtype=np.float32)
            nutilde = np.array(grid['Nu_Tilde'], dtype=np.float32)
            eddy_viscosity = np.array(grid['Eddy_Viscosity'], dtype=np.float32)

            velocities_2d = np.concatenate([raw_velocities[:, :2], raw_pressure[:, None], cp[:, None], cf[:, :2], Y_plus[:, None], nutilde[:, None], eddy_viscosity[:, None]], axis=1)
            
            # Check for consistent number of nodes
            if trajectory_numbers and coordinates.shape[0] != all_trajectory_data_raw[trajectory_numbers[0]]['coordinates'].shape[0]:
                print(f"Error: Inconsistent number of nodes in {file_path}. Expected {all_trajectory_data_raw[trajectory_numbers[0]]['coordinates'].shape[0]}, got {coordinates.shape[0]}. Skipping file.")
                continue

            trajectory_numbers.append(traj_num)
            flow_params_map[traj_num] = (reynolds, alpha)
            all_trajectory_data_raw[traj_num] = {
                'coordinates': coordinates,
                'edge_index': edge_index,
                'velocities': torch.tensor(velocities_2d, dtype=torch.float32)
            }
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
    
    trajectory_numbers.sort() # Ensure consistent order before stacking

    # Stack all velocities for scaling: [total_trajectories, num_nodes, num_velocity_components]
    stacked_velocities_list = [all_trajectory_data_raw[tn]['velocities'] for tn in trajectory_numbers]
    
    # Validate shapes before stacking
    first_tensor_shape = stacked_velocities_list[0].shape
    if not all(t.shape == first_tensor_shape for t in stacked_velocities_list):
        print("Error: Velocity tensors have inconsistent shapes across trajectories. Cannot stack for scaling.")
        return None, None, None
    
    num_velocity_components = first_tensor_shape[1]
    velocities_to_scale = torch.stack(stacked_velocities_list, dim=0) # Shape: [num_trajectories, num_nodes, num_components]

    # Split trajectory numbers for train/val sets
    train_traj_nums, val_traj_nums = split_dataset_trajectories(trajectory_numbers, train_ratio)

    scaler = None
    scaled_all_velocities_tensor = velocities_to_scale # Default if no scaling

    if not train_traj_nums:
        print("Warning: No training trajectories after splitting. Scaling will be skipped.")
    else:
        train_indices_in_stacked_tensor = [trajectory_numbers.index(tn) for tn in train_traj_nums]
        train_velocities_for_scaler = velocities_to_scale[train_indices_in_stacked_tensor]
        
        # Convert to numpy for scale_dataset function
        train_velocities_numpy = train_velocities_for_scaler.numpy()
        
        print(f"Fitting scaler on training data of shape: {train_velocities_numpy.shape}")
        
        # Use the scale_dataset function
        scaled_train_velocities, scaler = scale_dataset(
            train_velocities_numpy, 
            scaler=None, 
            method=scaling_method, 
            return_scaler=True
        )
        
        if scaler:
            print("Scaler fitted. Applying to the entire dataset.")
            # Convert all velocities to numpy for scaling
            all_velocities_numpy = velocities_to_scale.numpy()
            
            # Apply scaling to entire dataset
            scaled_all_velocities_numpy = scale_dataset(
                all_velocities_numpy, 
                scaler=scaler, 
                method=scaling_method, 
                return_scaler=False
            )
            
            # Convert back to tensor
            scaled_all_velocities_tensor = torch.tensor(scaled_all_velocities_numpy, dtype=torch.float32)
        else:
            print("Warning: Scaler fitting failed. Scaling will be skipped.")

    # Save the fitted scaler
    if scaler:
        with open(scaler_path, 'wb') as f_scaler:
            pickle.dump(scaler, f_scaler)
        print(f"Scaler saved to {scaler_path}")

    # Prepare data maps for writing (maps trajectory number to its data)
    all_scaled_velocities_map = {tn: scaled_all_velocities_tensor[trajectory_numbers.index(tn)] for tn in trajectory_numbers}
    all_coordinates_map = {tn: all_trajectory_data_raw[tn]['coordinates'] for tn in trajectory_numbers}

    # Write train H5 file
    print(f"Writing training data to {train_h5_path} ({len(train_traj_nums)} trajectories)")
    write_single_h5_file(train_h5_path, train_traj_nums, 
                         all_scaled_velocities_map, all_coordinates_map, edge_index, flow_params_map)
    
    # Write validation H5 file
    print(f"Writing validation data to {val_h5_path} ({len(val_traj_nums)} trajectories)")
    write_single_h5_file(val_h5_path, val_traj_nums, 
                         all_scaled_velocities_map, all_coordinates_map, edge_index, flow_params_map)

    print(f"VTU to H5 conversion with scaling and splitting complete. Output in {output_h5_dir}")
    return train_h5_path, val_h5_path, scaler_path