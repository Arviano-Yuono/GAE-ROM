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
        
        # Print summary of duplicates
        if duplicate_count > 0:
            print(f"📊 Summary: {duplicate_count} duplicate flow parameter combinations were found and handled.")
        else:
            print(f"📊 Summary: No duplicate flow parameters found.")
            

def rename_vtu_files(sorted_files, input_directory):
    """
    Rename VTU files to configuration_1.vtu, configuration_2.vtu, etc. based on their order in the sorted list.
    
    Args:
        sorted_files (list): List of sorted VTU filenames
        input_directory (str): Directory containing the VTU files
    
    Returns:
        list: List of new filenames in the order they were renamed
    """
    new_filenames = []
    
    for idx, old_filename in enumerate(sorted_files, 1):
        new_filename = f"configuration_{idx}.vtu"
        old_path = os.path.join(input_directory, old_filename)
        new_path = os.path.join(input_directory, new_filename)
        
        try:
            # Check if the new filename already exists
            if os.path.exists(new_path):
                print(f"Warning: {new_filename} already exists. Skipping rename.")
                continue
                
            # Rename the file
            os.rename(old_path, new_path)
            new_filenames.append(new_filename)
            #delete the old file
            print(f"Renamed {old_filename} to {new_filename}")
            os.remove(old_path)
            
        except Exception as e:
            print(f"Error renaming {old_filename}: {e}")
            continue
    
    return new_filenames

def vtu_to_h5(vtu_file_directory, 
              output_h5_dir, # Directory to save train.h5, val.h5, and scaler.pkl
              vtu_array_name='Velocity', # Name of the velocity array in VTU files
              train_ratio=0.9, 
              scaling_type=4, 
              scaler_name='standard',
              sorted_vtu_files=None,
              overwrite=False):
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

    vtu_files = sorted([f for f in os.listdir(vtu_file_directory) if f.lower().endswith('.vtu')])
    if not vtu_files:
        print(f"No VTU files found in {vtu_file_directory}")
        return None, None, None

    all_trajectory_data_raw = {} # Stores raw data: {traj_num: {'coordinates': ndarray, 'velocities': tensor}}
    trajectory_numbers = [] # List of integer trajectory numbers
    flow_params_map = {} # Dict: {traj_num: (reynolds, alpha)}

    print("Pass 1: Reading all VTU files and extracting data...")
    for vtu_file in tqdm.tqdm(vtu_files, desc="Reading VTUs"):
        try:
            # Extract flow parameters from filename or path
            # Try to extract from filename first (e.g., "Re4500000_alpha_10.vtu")
            base_name = os.path.splitext(vtu_file)[0]
            
            # Pattern to match flow_Re_XX_alpha_YY (with underscores around Re)
            import re
            pattern = r'flow_Re_(\d+(?:\.\d+)?)_alpha_(-?\d+(?:\.\d+)?)'
            match = re.search(pattern, base_name)
            
            if match:
                reynolds = float(match.group(1))
                alpha = float(match.group(2))
                traj_num = len(trajectory_numbers) + 1  # Use sequential numbering
                print(f"  ✅ Extracted: Re={reynolds:.1e}, alpha={alpha:.1f}° from {vtu_file}")
            else:
                # Fallback to old method
                traj_num_str = ''.join(filter(str.isdigit, base_name))
                if not traj_num_str:
                    print(f"Warning: Could not extract flow parameters from {vtu_file}. Skipping.")
                    continue
                
                traj_num = int(traj_num_str)
                # For backward compatibility, use default values
                reynolds = 1e6  # Default Reynolds number
                alpha = 0.0     # Default angle of attack
                print(f"  ⚠️ Using defaults: Re={reynolds:.1e}, alpha={alpha:.1f}° for {vtu_file}")
            
            grid = pv.UnstructuredGrid(os.path.join(vtu_file_directory, vtu_file))
            if grid is None or grid.points is None or vtu_array_name not in grid.point_data:
                 print(f"Warning: Failed to read {vtu_file} correctly or missing data. Skipping.")
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
                    print(f"Warning: Could not extract edge connectivity from {vtu_file}")
                    edge_index = None
            
            #velocity
            raw_velocities = np.array(grid.point_data[vtu_array_name], dtype=np.float32)
            surface_mask = raw_velocities[:,0] == 0.0
            raw_pressure = np.array(grid.point_data['Pressure'], dtype=np.float32)
            cp = np.array(grid['Pressure_Coefficient'], dtype=np.float32)
            cp[~surface_mask] = 0.0 # Set non-surface points to 0
            cf = np.array(grid['Skin_Friction_Coefficient'], dtype=np.float32)[:,:2]
            cf[~surface_mask] = 0.0 # Set non-surface points to 0

            velocities_2d = np.concat([raw_velocities[:, :2], raw_pressure[:, None], cp[:, None], cf[:, :2]], axis=1)
            
            # Check for consistent number of nodes
            if trajectory_numbers and coordinates.shape[0] != all_trajectory_data_raw[trajectory_numbers[0]]['coordinates'].shape[0]:
                print(f"Error: Inconsistent number of nodes in {vtu_file}. Expected {all_trajectory_data_raw[trajectory_numbers[0]]['coordinates'].shape[0]}, got {coordinates.shape[0]}. Skipping file.")
                continue

            trajectory_numbers.append(traj_num)
            flow_params_map[traj_num] = (reynolds, alpha)
            all_trajectory_data_raw[traj_num] = {
                'coordinates': coordinates,
                'edge_index': edge_index,
                'velocities': torch.tensor(velocities_2d, dtype=torch.float32)
            }
        except Exception as e:
            print(f"Error processing file {vtu_file}: {e}. Skipping.")
            continue
            
    if not trajectory_numbers:
        print("No valid trajectory data could be extracted after Pass 1.")
        return None, None, None
    
    # Print summary of flow parameters
    print(f"\n📊 Flow Parameters Summary:")
    unique_params = set(flow_params_map.values())
    for reynolds, alpha in sorted(unique_params):
        count = sum(1 for params in flow_params_map.values() if params == (reynolds, alpha))
        print(f"  Re={reynolds:.1e}, alpha={alpha:.1f}°: {count} files")
    
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
            if isinstance(scaler, list):
                # Handle case where scaler is a list (e.g., for scaling_type 3 or 4)
                print("Warning: Scaler is a list. Using first scaler for transformation.")
                if len(scaler) > 0 and hasattr(scaler[0], 'transform'):
                    scaled_transformed_data = scaler[0].transform(reshaped_all_velocities.numpy())
                    scaled_all_velocities_tensor = torch.tensor(scaled_transformed_data, dtype=torch.float32).reshape(original_shape_all)
                else:
                    print("Warning: First scaler in list does not have 'transform' method. Skipping scaling.")
                    scaled_all_velocities_tensor = velocities_to_scale
            elif hasattr(scaler, 'transform'):
                scaled_transformed_data = scaler.transform(reshaped_all_velocities.numpy())
                scaled_all_velocities_tensor = torch.tensor(scaled_transformed_data, dtype=torch.float32).reshape(original_shape_all)
            else:
                print("Warning: Fitted scaler does not have a 'transform' method. Scaling might not be applied correctly. Check 'src.data.scaling' module.")
                # Fallback or error based on how scaling module should behave
                scaled_all_velocities_tensor = velocities_to_scale
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
