import h5py
import numpy as np
import torch
import pickle
from src.data import scaling

def load_scalers(scaler_path):
    """
    Load the fitted scalers from the pickle file.
    
    Args:
        scaler_path: Path to the scaler.pkl file
    
    Returns:
        dict: Dictionary of scalers for each variable
    """
    with open(scaler_path, 'rb') as f:
        scalers = pickle.load(f)
    return scalers

def inverse_scale_single_trajectory(data_tensor, scaler, scaling_type):
    """
    Apply inverse scaling to a single trajectory tensor.
    
    Args:
        data_tensor: Tensor to inverse scale [num_nodes, num_features] or [num_nodes]
        scaler: Fitted scaler for the variable
        scaling_type: Type of scaling used (1, 2, 3, or 4)
    
    Returns:
        torch.Tensor: Inverse scaled tensor
    """
    # Ensure data is 2D
    if data_tensor.dim() == 1:
        data_tensor = data_tensor.unsqueeze(1)
    
    # Apply inverse scaling based on scaling type
    if scaling_type == 1:  # SAMPLE SCALING
        inverse_scaled = torch.tensor(scaler.inverse_transform(data_tensor.numpy()), dtype=torch.float32)
    elif scaling_type == 2:  # FEATURE SCALING
        inverse_scaled = torch.tensor(scaler.inverse_transform(data_tensor.T).T, dtype=torch.float32)
    elif scaling_type == 3:  # FEATURE-SAMPLE SCALING
        scaler_f, scaler_s = scaler
        # Inverse the order: first inverse scaler_s, then inverse scaler_f
        temp = scaler_s.inverse_transform(data_tensor.T)
        inverse_scaled = torch.tensor(scaler_f.inverse_transform(temp).T, dtype=torch.float32)
    elif scaling_type == 4:  # SAMPLE-FEATURE SCALING
        scaler_s, scaler_f = scaler
        # Inverse the order: first inverse scaler_f, then inverse scaler_s
        temp = scaler_f.inverse_transform(data_tensor.T)
        inverse_scaled = torch.tensor(scaler_s.inverse_transform(temp).T, dtype=torch.float32)
    else:
        print(f"Warning: Unknown scaling type {scaling_type}")
        inverse_scaled = data_tensor
    
    return inverse_scaled

def inverse_scale_batch_trajectories(data_tensor, scaler, scaling_type):
    """
    Apply inverse scaling to a batch of trajectories.
    
    Args:
        data_tensor: Tensor to inverse scale [num_trajectories, num_nodes, num_features]
        scaler: Fitted scaler for the variable
        scaling_type: Type of scaling used (1, 2, 3, or 4)
    
    Returns:
        torch.Tensor: Inverse scaled tensor
    """
    original_shape = data_tensor.shape
    
    # Reshape for inverse scaling (same as in vtu_to_h5.py)
    reshaped_data = data_tensor.reshape(-1, data_tensor.shape[-1])
    
    # Apply inverse scaling
    if scaling_type == 1:  # SAMPLE SCALING
        inverse_scaled_reshaped = torch.tensor(scaler.inverse_transform(reshaped_data.numpy()), dtype=torch.float32)
    elif scaling_type == 2:  # FEATURE SCALING
        inverse_scaled_reshaped = torch.tensor(scaler.inverse_transform(reshaped_data.T).T, dtype=torch.float32)
    elif scaling_type == 3:  # FEATURE-SAMPLE SCALING
        scaler_f, scaler_s = scaler
        temp = scaler_s.inverse_transform(reshaped_data.T)
        inverse_scaled_reshaped = torch.tensor(scaler_f.inverse_transform(temp).T, dtype=torch.float32)
    elif scaling_type == 4:  # SAMPLE-FEATURE SCALING
        scaler_s, scaler_f = scaler
        temp = scaler_f.inverse_transform(reshaped_data.T)
        inverse_scaled_reshaped = torch.tensor(scaler_s.inverse_transform(temp).T, dtype=torch.float32)
    else:
        print(f"Warning: Unknown scaling type {scaling_type}")
        inverse_scaled_reshaped = reshaped_data
    
    # Reshape back to original shape
    return inverse_scaled_reshaped.reshape(original_shape)

def inverse_scale_h5_dataset(h5_file_path, variable_name, scaler, scaling_type, output_path=None):
    """
    Apply inverse scaling to a specific variable in an H5 file.
    
    Args:
        h5_file_path: Path to the input H5 file (containing scaled data)
        variable_name: Name of the variable to inverse scale (e.g., 'Ux', 'Uy', 'Pressure')
        scaler: Fitted scaler for the variable
        scaling_type: Type of scaling used (1, 2, 3, or 4)
        output_path: Path for the output H5 file (if None, overwrites input file)
    
    Returns:
        str: Path to the output H5 file
    """
    if output_path is None:
        output_path = h5_file_path
    
    # Collect all data for the target variable across all trajectories
    print(f"Collecting {variable_name} data from all trajectories...")
    all_data = []
    trajectory_names = []
    
    with h5py.File(h5_file_path, 'r') as f_in:
        trajectory_names = list(f_in.keys())
        
        for traj_name in trajectory_names:
            if variable_name in f_in[traj_name]:
                data = f_in[traj_name][variable_name][:]
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                all_data.append(torch.tensor(data, dtype=torch.float32))
    
    # Apply inverse scaling using the same approach as scaling
    print(f"Applying inverse scaling to {variable_name}...")
    
    # Stack all trajectories for this variable
    stacked_data = torch.stack(all_data, dim=0)  # [num_trajectories, num_nodes, num_features]
    
    # Apply inverse scaling
    inverse_scaled_data = inverse_scale_batch_trajectories(stacked_data, scaler, scaling_type)
    
    # Split back into individual trajectories
    inverse_scaled_trajectories = [inverse_scaled_data[i] for i in range(inverse_scaled_data.shape[0])]
    
    # Write the inverse scaled data back to H5 file
    print("Writing inverse scaled data to H5 file...")
    with h5py.File(h5_file_path, 'r') as f_in, h5py.File(output_path, 'w') as f_out:
        for traj_idx, traj_name in enumerate(trajectory_names):
            f_out.create_group(traj_name)
            
            for dataset_name in f_in[traj_name].keys():
                if dataset_name == variable_name:
                    # Use inverse scaled data
                    inverse_scaled_traj_data = inverse_scaled_trajectories[traj_idx]
                    f_out[traj_name][dataset_name] = inverse_scaled_traj_data.numpy()
                else:
                    # Copy other datasets as-is
                    f_out[traj_name][dataset_name] = f_in[traj_name][dataset_name][:]
    
    return output_path

def inverse_scale_all_variables(h5_file_path, scalers, scaling_type, output_path=None):
    """
    Apply inverse scaling to all variables in an H5 file.
    
    Args:
        h5_file_path: Path to the input H5 file (containing scaled data)
        scalers: Dictionary of scalers for each variable
        scaling_type: Type of scaling used (1, 2, 3, or 4)
        output_path: Path for the output H5 file (if None, overwrites input file)
    
    Returns:
        str: Path to the output H5 file
    """
    if output_path is None:
        output_path = h5_file_path
    
    # First, collect all data for each variable across all trajectories
    print("Collecting data from all trajectories...")
    all_data_by_variable = {}
    trajectory_names = []
    
    with h5py.File(h5_file_path, 'r') as f_in:
        trajectory_names = list(f_in.keys())
        
        for var_name in scalers.keys():
            all_data_by_variable[var_name] = []
            
            for traj_name in trajectory_names:
                if var_name in f_in[traj_name]:
                    data = f_in[traj_name][var_name][:]
                    if data.ndim == 1:
                        data = data.reshape(-1, 1)
                    all_data_by_variable[var_name].append(torch.tensor(data, dtype=torch.float32))
    
    # Apply inverse scaling to each variable
    print("Applying inverse scaling to each variable...")
    inverse_scaled_data_by_variable = {}
    
    for var_name, data_list in all_data_by_variable.items():
        print(f"Inverse scaling {var_name}...")
        
        # Stack all trajectories for this variable
        stacked_data = torch.stack(data_list, dim=0)  # [num_trajectories, num_nodes, num_features]
        
        # Apply inverse scaling using the fitted scaler
        scaler = scalers[var_name]
        inverse_scaled_data = inverse_scale_batch_trajectories(stacked_data, scaler, scaling_type)
        
        # Split back into individual trajectories
        inverse_scaled_data_by_variable[var_name] = [inverse_scaled_data[i] for i in range(inverse_scaled_data.shape[0])]
    
    # Write the inverse scaled data back to H5 file
    print("Writing inverse scaled data to H5 file...")
    with h5py.File(h5_file_path, 'r') as f_in, h5py.File(output_path, 'w') as f_out:
        for traj_idx, traj_name in enumerate(trajectory_names):
            f_out.create_group(traj_name)
            
            for dataset_name in f_in[traj_name].keys():
                if dataset_name in scalers:
                    # Use inverse scaled data
                    inverse_scaled_traj_data = inverse_scaled_data_by_variable[dataset_name][traj_idx]
                    f_out[traj_name][dataset_name] = inverse_scaled_traj_data.numpy()
                else:
                    # Copy other datasets as-is
                    f_out[traj_name][dataset_name] = f_in[traj_name][dataset_name][:]
    
    return output_path

def apply_scaling_to_h5_dataset(h5_file_path, variable_name, scaler, scaling_type, output_path=None):
    """
    Apply scaling to a specific variable in an H5 file.
    
    Args:
        h5_file_path: Path to the input H5 file
        variable_name: Name of the variable to scale (e.g., 'Ux', 'Uy', 'Pressure')
        scaler: Fitted scaler for the variable
        scaling_type: Type of scaling used (1, 2, 3, or 4)
        output_path: Path for the output H5 file (if None, overwrites input file)
    
    Returns:
        str: Path to the output H5 file
    """
    if output_path is None:
        output_path = h5_file_path
    
    # Collect all data for the target variable across all trajectories
    print(f"Collecting {variable_name} data from all trajectories...")
    all_data = []
    trajectory_names = []
    
    with h5py.File(h5_file_path, 'r') as f_in:
        trajectory_names = list(f_in.keys())
        
        for traj_name in trajectory_names:
            if variable_name in f_in[traj_name]:
                data = f_in[traj_name][variable_name][:]
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                all_data.append(torch.tensor(data, dtype=torch.float32))
    
    # Apply scaling using the same approach as vtu_to_h5.py
    print(f"Applying scaling to {variable_name}...")
    
    # Stack all trajectories for this variable
    stacked_data = torch.stack(all_data, dim=0)  # [num_trajectories, num_nodes, num_features]
    original_shape = stacked_data.shape
    
    # Reshape for scaling (same as in vtu_to_h5.py)
    reshaped_data = stacked_data.reshape(-1, stacked_data.shape[-1])
    
    # Apply scaling using the fitted scaler
    if scaling_type == 1:  # SAMPLE SCALING
        scaled_reshaped = torch.tensor(scaler.transform(reshaped_data.numpy()), dtype=torch.float32)
    elif scaling_type == 2:  # FEATURE SCALING
        scaled_reshaped = torch.tensor(scaler.transform(reshaped_data.T).T, dtype=torch.float32)
    elif scaling_type == 3:  # FEATURE-SAMPLE SCALING
        scaler_f, scaler_s = scaler
        temp = scaler_f.transform(reshaped_data.T)
        scaled_reshaped = torch.tensor(scaler_s.transform(temp).T, dtype=torch.float32)
    elif scaling_type == 4:  # SAMPLE-FEATURE SCALING
        scaler_s, scaler_f = scaler
        temp = scaler_s.transform(reshaped_data)
        scaled_reshaped = torch.tensor(scaler_f.transform(temp.T).T, dtype=torch.float32)
    else:
        print(f"Warning: Unknown scaling type {scaling_type}")
        scaled_reshaped = reshaped_data
    
    # Reshape back to original shape
    scaled_data = scaled_reshaped.reshape(original_shape)
    
    # Split back into individual trajectories
    scaled_trajectories = [scaled_data[i] for i in range(scaled_data.shape[0])]
    
    # Write the scaled data back to H5 file
    print("Writing scaled data to H5 file...")
    with h5py.File(h5_file_path, 'r') as f_in, h5py.File(output_path, 'w') as f_out:
        for traj_idx, traj_name in enumerate(trajectory_names):
            f_out.create_group(traj_name)
            
            for dataset_name in f_in[traj_name].keys():
                if dataset_name == variable_name:
                    # Use scaled data
                    scaled_traj_data = scaled_trajectories[traj_idx]
                    f_out[traj_name][dataset_name] = scaled_traj_data.numpy()
                else:
                    # Copy other datasets as-is
                    f_out[traj_name][dataset_name] = f_in[traj_name][dataset_name][:]
    
    return output_path

def apply_scaling_to_all_variables(h5_file_path, scalers, scaling_type, output_path=None):
    """
    Apply scaling to all variables in an H5 file.
    
    Args:
        h5_file_path: Path to the input H5 file
        scalers: Dictionary of scalers for each variable
        scaling_type: Type of scaling used (1, 2, 3, or 4)
        output_path: Path for the output H5 file (if None, overwrites input file)
    
    Returns:
        str: Path to the output H5 file
    """
    if output_path is None:
        output_path = h5_file_path
    
    # First, collect all data for each variable across all trajectories
    print("Collecting data from all trajectories...")
    all_data_by_variable = {}
    trajectory_names = []
    
    with h5py.File(h5_file_path, 'r') as f_in:
        trajectory_names = list(f_in.keys())
        
        for var_name in scalers.keys():
            all_data_by_variable[var_name] = []
            
            for traj_name in trajectory_names:
                if var_name in f_in[traj_name]:
                    data = f_in[traj_name][var_name][:]
                    if data.ndim == 1:
                        data = data.reshape(-1, 1)
                    all_data_by_variable[var_name].append(torch.tensor(data, dtype=torch.float32))
    
    # Apply scaling to each variable using the same approach as vtu_to_h5.py
    print("Applying scaling to each variable...")
    scaled_data_by_variable = {}
    
    for var_name, data_list in all_data_by_variable.items():
        print(f"Scaling {var_name}...")
        
        # Stack all trajectories for this variable
        stacked_data = torch.stack(data_list, dim=0)  # [num_trajectories, num_nodes, num_features]
        original_shape = stacked_data.shape
        
        # Reshape for scaling (same as in vtu_to_h5.py)
        reshaped_data = stacked_data.reshape(-1, stacked_data.shape[-1])
        
        # Apply scaling using the fitted scaler
        scaler = scalers[var_name]
        
        if scaling_type == 1:  # SAMPLE SCALING
            scaled_reshaped = torch.tensor(scaler.transform(reshaped_data.numpy()), dtype=torch.float32)
        elif scaling_type == 2:  # FEATURE SCALING
            scaled_reshaped = torch.tensor(scaler.transform(reshaped_data.T).T, dtype=torch.float32)
        elif scaling_type == 3:  # FEATURE-SAMPLE SCALING
            scaler_f, scaler_s = scaler
            temp = scaler_f.transform(reshaped_data.T)
            scaled_reshaped = torch.tensor(scaler_s.transform(temp).T, dtype=torch.float32)
        elif scaling_type == 4:  # SAMPLE-FEATURE SCALING
            scaler_s, scaler_f = scaler
            temp = scaler_s.transform(reshaped_data)
            scaled_reshaped = torch.tensor(scaler_f.transform(temp.T).T, dtype=torch.float32)
        else:
            print(f"Warning: Unknown scaling type {scaling_type}")
            scaled_reshaped = reshaped_data
        
        # Reshape back to original shape
        scaled_data = scaled_reshaped.reshape(original_shape)
        
        # Split back into individual trajectories
        scaled_data_by_variable[var_name] = [scaled_data[i] for i in range(scaled_data.shape[0])]
    
    # Write the scaled data back to H5 file
    print("Writing scaled data to H5 file...")
    with h5py.File(h5_file_path, 'r') as f_in, h5py.File(output_path, 'w') as f_out:
        for traj_idx, traj_name in enumerate(trajectory_names):
            f_out.create_group(traj_name)
            
            for dataset_name in f_in[traj_name].keys():
                if dataset_name in scalers:
                    # Use scaled data
                    scaled_traj_data = scaled_data_by_variable[dataset_name][traj_idx]
                    f_out[traj_name][dataset_name] = scaled_traj_data.numpy()
                else:
                    # Copy other datasets as-is
                    f_out[traj_name][dataset_name] = f_in[traj_name][dataset_name][:]
    
    return output_path

def get_variable_statistics(h5_file_path, variable_name):
    """
    Get basic statistics for a variable in an H5 file.
    
    Args:
        h5_file_path: Path to the H5 file
        variable_name: Name of the variable
    
    Returns:
        dict: Statistics including min, max, mean, std
    """
    with h5py.File(h5_file_path, 'r') as f:
        all_data = []
        for group_name in f.keys():
            if variable_name in f[group_name]:
                data = f[group_name][variable_name][:]
                all_data.append(data.flatten())
        
        if all_data:
            combined_data = np.concatenate(all_data)
            return {
                'min': np.min(combined_data),
                'max': np.max(combined_data),
                'mean': np.mean(combined_data),
                'std': np.std(combined_data),
                'shape': combined_data.shape
            }
        else:
            return None

# Example usage functions
def scale_ux_dataset_example():
    """
    Example of how to apply scaling to a Ux dataset.
    """
    # Paths
    scaler_path = './dataset/Ux_h5_files/scaler.pkl'
    h5_file_path = './dataset/Ux_h5_files/train.h5'
    output_path = './dataset/Ux_h5_files/train_scaled.h5'
    scaling_type = 4  # SAMPLE-FEATURE SCALING
    
    # Load scalers
    print("Loading scalers...")
    scalers = load_scalers(scaler_path)
    print(f"Available scalers: {list(scalers.keys())}")
    
    # Get statistics before scaling
    print("\nStatistics before scaling:")
    stats_before = get_variable_statistics(h5_file_path, 'Ux')
    if stats_before:
        print(f"Ux - Min: {stats_before['min']:.6f}, Max: {stats_before['max']:.6f}")
        print(f"     Mean: {stats_before['mean']:.6f}, Std: {stats_before['std']:.6f}")
    
    # Apply scaling to Ux only
    print("\nApplying scaling to Ux...")
    apply_scaling_to_h5_dataset(h5_file_path, 'Ux', scalers['Ux'], scaling_type, output_path)
    
    # Get statistics after scaling
    print("\nStatistics after scaling:")
    stats_after = get_variable_statistics(output_path, 'Ux')
    if stats_after:
        print(f"Ux - Min: {stats_after['min']:.6f}, Max: {stats_after['max']:.6f}")
        print(f"     Mean: {stats_after['mean']:.6f}, Std: {stats_after['std']:.6f}")
    
    print(f"\nScaled data saved to: {output_path}")

def scale_all_variables_example():
    """
    Example of how to apply scaling to all variables in a dataset.
    """
    # Paths
    scaler_path = './dataset/Ux_h5_files/scaler.pkl'
    h5_file_path = './dataset/Ux_h5_files/train.h5'
    output_path = './dataset/Ux_h5_files/train_all_scaled.h5'
    scaling_type = 4  # SAMPLE-FEATURE SCALING
    
    # Load scalers
    print("Loading scalers...")
    scalers = load_scalers(scaler_path)
    print(f"Available scalers: {list(scalers.keys())}")
    
    # Apply scaling to all variables
    print("\nApplying scaling to all variables...")
    apply_scaling_to_all_variables(h5_file_path, scalers, scaling_type, output_path)
    
    print(f"\nAll scaled data saved to: {output_path}")

def inverse_scale_example():
    """
    Example of how to apply inverse scaling to scaled data.
    """
    # Paths
    scaler_path = './dataset/new_h5_files/scaler.pkl'
    scaled_h5_file = './dataset/new_h5_files/train_scaled.h5'
    output_path = './dataset/new_h5_files/train_inverse_scaled.h5'
    scaling_type = 4  # SAMPLE-FEATURE SCALING
    
    # Load scalers
    print("Loading scalers...")
    scalers = load_scalers(scaler_path)
    print(f"Available scalers: {list(scalers.keys())}")
    
    # Get statistics before inverse scaling
    print("\nStatistics before inverse scaling:")
    stats_before = get_variable_statistics(scaled_h5_file, 'Ux')
    if stats_before:
        print(f"Ux - Min: {stats_before['min']:.6f}, Max: {stats_before['max']:.6f}")
        print(f"     Mean: {stats_before['mean']:.6f}, Std: {stats_before['std']:.6f}")
    
    # Apply inverse scaling to all variables
    print("\nApplying inverse scaling to all variables...")
    inverse_scale_all_variables(scaled_h5_file, scalers, scaling_type, output_path)
    
    # Get statistics after inverse scaling
    print("\nStatistics after inverse scaling:")
    stats_after = get_variable_statistics(output_path, 'Ux')
    if stats_after:
        print(f"Ux - Min: {stats_after['min']:.6f}, Max: {stats_after['max']:.6f}")
        print(f"     Mean: {stats_after['mean']:.6f}, Std: {stats_after['std']:.6f}")
    
    print(f"\nInverse scaled data saved to: {output_path}")

def single_trajectory_inverse_scale_example():
    """
    Example of how to inverse scale a single trajectory tensor.
    """
    # Load scalers
    scalers = load_scalers('./dataset/new_h5_files/scaler.pkl')
    scaling_type = 4
    
    # Create a dummy scaled trajectory (this would come from your model prediction)
    # Shape: [num_nodes, num_features] for single feature or [num_nodes, 2] for 2 features
    dummy_scaled_trajectory = torch.randn(14576, 1)  # Example: 14576 nodes, 1 feature (Ux)
    
    print("Original scaled trajectory statistics:")
    print(f"Min: {dummy_scaled_trajectory.min():.6f}, Max: {dummy_scaled_trajectory.max():.6f}")
    print(f"Mean: {dummy_scaled_trajectory.mean():.6f}, Std: {dummy_scaled_trajectory.std():.6f}")
    
    # Apply inverse scaling
    inverse_scaled_trajectory = inverse_scale_single_trajectory(
        dummy_scaled_trajectory, 
        scalers['Ux'], 
        scaling_type
    )
    
    print("\nInverse scaled trajectory statistics:")
    print(f"Min: {inverse_scaled_trajectory.min():.6f}, Max: {inverse_scaled_trajectory.max():.6f}")
    print(f"Mean: {inverse_scaled_trajectory.mean():.6f}, Std: {inverse_scaled_trajectory.std():.6f}")
    
    return inverse_scaled_trajectory

if __name__ == "__main__":
    # Run examples
    print("=== Scaling Ux Dataset Example ===")
    scale_ux_dataset_example()
    
    print("\n=== Scaling All Variables Example ===")
    scale_all_variables_example()
    
    print("\n=== Inverse Scaling Example ===")
    inverse_scale_example()
    
    print("\n=== Single Trajectory Inverse Scaling Example ===")
    single_trajectory_inverse_scale_example() 