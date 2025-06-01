import h5py
import pyvista as pv
import os
import tqdm
import numpy as np
import torch
import pickle
from src.data import scaling # Ensure this import is correct and module available

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
                         edge_index):
    with h5py.File(output_h5_path, 'w') as f:
        for traj_num in tqdm.tqdm(trajectories_to_write, desc=f"Writing {os.path.basename(output_h5_path)}"):
            g = f.create_group(f'configuration_{traj_num}')
            
            g['coordinates'] = all_coordinates_map[traj_num]
            if edge_index is not None:
                g['edge_index'] = edge_index
            
            velocities_for_traj = all_scaled_velocities_map[traj_num] # Should be [num_nodes, 2]
            g['Ux'] = velocities_for_traj[:, 0].numpy() # Convert to numpy for h5py
            g['Uy'] = velocities_for_traj[:, 1].numpy() # Convert to numpy for h5py

def vtu_to_h5(vtu_file_directory, 
              output_h5_dir, # Directory to save train.h5, val.h5, and scaler.pkl
              vtu_array_name='Velocity', # Name of the velocity array in VTU files
              train_ratio=0.8, 
              scaling_type=4, 
              scaler_name='standard', 
              edge_index=None, # Assumed to be constant for the dataset
              overwrite=False):

    train_h5_path = os.path.join(output_h5_dir, 'train.h5')
    val_h5_path = os.path.join(output_h5_dir, 'val.h5')
    scaler_path = os.path.join(output_h5_dir, 'scaler.pkl')

    if not os.path.exists(output_h5_dir):
        os.makedirs(output_h5_dir)
    
    if not overwrite:
        if os.path.exists(train_h5_path) or os.path.exists(val_h5_path):
            print(f"H5 files in {output_h5_dir} already exist. Set overwrite=True to overwrite.")
            return None, None, None

    vtu_files = sorted([f for f in os.listdir(vtu_file_directory) if f.lower().endswith('.vtu')])
    if not vtu_files:
        print(f"No VTU files found in {vtu_file_directory}")
        return None, None, None

    all_trajectory_data_raw = {} # Stores raw data: {traj_num: {'coordinates': ndarray, 'velocities': tensor}}
    trajectory_numbers = [] # List of integer trajectory numbers

    print("Pass 1: Reading all VTU files and extracting data...")
    for vtu_file in tqdm.tqdm(vtu_files, desc="Reading VTUs"):
        try:
            base_name = os.path.splitext(vtu_file)[0] 
            traj_num_str = ''.join(filter(str.isdigit, base_name))
            if not traj_num_str:
                print(f"Warning: Could not extract trajectory number from {vtu_file}. Using index as fallback or skipping.")
                # Fallback to using index if no number found, or skip
                # For this example, let's try to use a simple counter if parsing fails, or skip
                # traj_num = len(trajectory_numbers) + 1 # Example fallback
                print(f"Skipping {vtu_file} as trajectory number extraction failed.")
                continue # Skipping if number extraction is crucial and fails

            traj_num = int(traj_num_str)
            
            grid = pv.read(os.path.join(vtu_file_directory, vtu_file))
            if grid is None or grid.points is None or vtu_array_name not in grid.point_data:
                 print(f"Warning: Failed to read {vtu_file} correctly or missing data. Skipping.")
                 continue

            coordinates = np.array(grid.points[:, 0:2], dtype=np.float32) 
            
            raw_velocities = np.array(grid.point_data[vtu_array_name], dtype=np.float32)
            if raw_velocities.ndim == 1: # Case of single component array per node
                if raw_velocities.shape[0] == coordinates.shape[0] * 2 : # Might be flattened UxUyUxUy
                    velocities_2d = raw_velocities.reshape(coordinates.shape[0], 2)
                else: # Assume it's a single component (e.g. pressure, or Ux only)
                     print(f"Warning: Velocity array in {vtu_file} is 1D with unexpected shape {raw_velocities.shape}. Trying to use as single component.")
                     velocities_2d = raw_velocities.reshape(-1,1) # Reshape to [N,1]
            elif raw_velocities.shape[1] == 3: # [N, 3] for (vx, vy, vz)
                velocities_2d = raw_velocities[:, 0:2]
            elif raw_velocities.shape[1] == 2: # [N, 2] for (vx, vy)
                velocities_2d = raw_velocities
            elif raw_velocities.shape[1] == 1: # [N, 1] e.g. Ux only
                velocities_2d = raw_velocities
            else:
                print(f"Warning: Unexpected velocity shape {raw_velocities.shape} in {vtu_file}. Expected 1, 2 or 3 components. Skipping.")
                continue
            
            # Check for consistent number of nodes
            if trajectory_numbers and coordinates.shape[0] != all_trajectory_data_raw[trajectory_numbers[0]]['coordinates'].shape[0]:
                print(f"Error: Inconsistent number of nodes in {vtu_file}. Expected {all_trajectory_data_raw[trajectory_numbers[0]]['coordinates'].shape[0]}, got {coordinates.shape[0]}. Skipping file.")
                continue

            trajectory_numbers.append(traj_num)
            all_trajectory_data_raw[traj_num] = {
                'coordinates': coordinates,
                'velocities': torch.tensor(velocities_2d, dtype=torch.float32)
            }
        except Exception as e:
            print(f"Error processing file {vtu_file}: {e}. Skipping.")
            continue
            
    if not trajectory_numbers:
        print("No valid trajectory data could be extracted after Pass 1.")
        return None, None, None
    
    trajectory_numbers.sort() # Ensure consistent order before stacking

    # Stack all velocities for scaling: [total_trajectories, num_nodes, num_velocity_components]
    # num_velocity_components can be 1 or 2
    stacked_velocities_list = [all_trajectory_data_raw[tn]['velocities'] for tn in trajectory_numbers]
    
    # Validate shapes before stacking
    first_tensor_shape = stacked_velocities_list[0].shape
    if not all(t.shape == first_tensor_shape for t in stacked_velocities_list):
        print("Error: Velocity tensors have inconsistent shapes across trajectories. Cannot stack for scaling.")
        # Debug inconsistent shapes:
        # for i, tn in enumerate(trajectory_numbers):
        # print(f"Traj {tn}, shape: {all_trajectory_data_raw[tn]['velocities'].shape}")
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
        
        original_shape_train = train_velocities_for_scaler.shape # [num_train_traj, num_nodes, num_components]
        reshaped_train_velocities = train_velocities_for_scaler.reshape(-1, num_velocity_components) # [N_train_samples * num_nodes, num_components]
        
        print(f"Fitting scaler on training data of shape: {reshaped_train_velocities.shape}")
        scaler, _ = scaling.tensor_scaling(reshaped_train_velocities, scaling_type, scaler_name) # Fit scaler
        
        if scaler:
            print("Scaler fitted. Applying to the entire dataset.")
            original_shape_all = velocities_to_scale.shape # [total_traj, num_nodes, num_components]
            reshaped_all_velocities = velocities_to_scale.reshape(-1, num_velocity_components)
            
            # Use the transform method of the fitted scaler
            if hasattr(scaler, 'transform'):
                scaled_transformed_data = scaler.transform(reshaped_all_velocities.numpy())
                scaled_all_velocities_tensor = torch.tensor(scaled_transformed_data, dtype=torch.float32).reshape(original_shape_all)
            else:
                print("Warning: Fitted scaler does not have a 'transform' method. Scaling might not be applied correctly. Check 'src.data.scaling' module.")
                # Fallback or error based on how scaling module should behave
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
                         all_scaled_velocities_map, all_coordinates_map, edge_index)
    
    # Write validation H5 file
    print(f"Writing validation data to {val_h5_path} ({len(val_traj_nums)} trajectories)")
    write_single_h5_file(val_h5_path, val_traj_nums, 
                         all_scaled_velocities_map, all_coordinates_map, edge_index)

    print(f"VTU to H5 conversion with scaling and splitting complete. Output in {output_h5_dir}")
    return train_h5_path, val_h5_path, scaler_path
