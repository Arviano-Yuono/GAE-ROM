import h5py
import numpy as np
import torch
import pickle
from src.data.apply_scaling import load_scalers

def test_scaling_consistency():
    """
    Test if the scaling works correctly with the actual data.
    """
    # Load scalers
    scaler_path = './dataset/new_h5_files/scaler.pkl'
    scalers = load_scalers(scaler_path)
    
    print("Available scalers:", list(scalers.keys()))
    
    # Check what the scalers expect
    for var_name, scaler in scalers.items():
        if hasattr(scaler, 'n_features_in_'):
            print(f"{var_name} scaler expects {scaler.n_features_in_} features")
        else:
            print(f"{var_name} scaler is a list of scalers")
    
    # Test with actual data
    train_file = './dataset/new_h5_files/train.h5'
    val_file = './dataset/new_h5_files/val_unscaled.h5'
    
    # Check a few trajectories from each file
    print("\n=== Checking Training Data ===")
    with h5py.File(train_file, 'r') as f:
        for i, group_name in enumerate(list(f.keys())[:3]):  # Check first 3
            if 'Ux' in f[group_name]:
                data = f[group_name]['Ux'][:]
                print(f"  {group_name}: {data.shape}")
    
    print("\n=== Checking Validation Data ===")
    with h5py.File(val_file, 'r') as f:
        for i, group_name in enumerate(list(f.keys())[:3]):  # Check first 3
            if 'Ux' in f[group_name]:
                data = f[group_name]['Ux'][:]
                print(f"  {group_name}: {data.shape}")
    
    # Test scaling on a single trajectory
    print("\n=== Testing Single Trajectory Scaling ===")
    with h5py.File(train_file, 'r') as f:
        # Get first trajectory
        first_group = list(f.keys())[0]
        ux_data = f[first_group]['Ux'][:]
        print(f"Original Ux data shape: {ux_data.shape}")
        print(f"Original Ux range: [{ux_data.min():.6f}, {ux_data.max():.6f}]")
        
        # Convert to tensor
        ux_tensor = torch.tensor(ux_data, dtype=torch.float32).unsqueeze(1)  # [num_nodes, 1]
        
        # Test scaling
        scaler = scalers['Ux']
        scaling_type = 4
        
        try:
            if scaling_type == 4:  # SAMPLE-FEATURE SCALING
                scaler_s, scaler_f = scaler
                temp = scaler_s.transform(ux_tensor)
                scaled_data = torch.tensor(scaler_f.transform(temp.T).T, dtype=torch.float32)
                print(f"Scaled Ux range: [{scaled_data.min():.6f}, {scaled_data.max():.6f}]")
                print("✅ Single trajectory scaling works!")
            else:
                print("❌ Unexpected scaling type")
        except Exception as e:
            print(f"❌ Error in single trajectory scaling: {e}")

def test_batch_scaling():
    """
    Test batch scaling with multiple trajectories.
    """
    print("\n=== Testing Batch Scaling ===")
    
    # Load scalers
    scaler_path = './dataset/new_h5_files/scaler.pkl'
    scalers = load_scalers(scaler_path)
    
    # Get data from training file
    train_file = './dataset/new_h5_files/train.h5'
    
    with h5py.File(train_file, 'r') as f:
        # Get first 3 trajectories
        trajectories = []
        for i, group_name in enumerate(list(f.keys())[:3]):
            if 'Ux' in f[group_name]:
                data = f[group_name]['Ux'][:]
                trajectories.append(torch.tensor(data, dtype=torch.float32).unsqueeze(1))
        
        if trajectories:
            # Stack trajectories
            stacked_data = torch.stack(trajectories, dim=0)  # [3, num_nodes, 1]
            print(f"Stacked data shape: {stacked_data.shape}")
            
            # Reshape for scaling
            reshaped_data = stacked_data.reshape(-1, stacked_data.shape[-1])
            print(f"Reshaped data shape: {reshaped_data.shape}")
            
            # Apply scaling
            scaler = scalers['Ux']
            scaling_type = 4
            
            try:
                if scaling_type == 4:  # SAMPLE-FEATURE SCALING
                    scaler_s, scaler_f = scaler
                    temp = scaler_s.transform(reshaped_data)
                    scaled_reshaped = torch.tensor(scaler_f.transform(temp.T).T, dtype=torch.float32)
                    
                    # Reshape back
                    scaled_data = scaled_reshaped.reshape(stacked_data.shape)
                    print(f"Scaled data shape: {scaled_data.shape}")
                    print(f"Scaled data range: [{scaled_data.min():.6f}, {scaled_data.max():.6f}]")
                    print("✅ Batch scaling works!")
                    
                else:
                    print("❌ Unexpected scaling type")
            except Exception as e:
                print(f"❌ Error in batch scaling: {e}")

if __name__ == "__main__":
    test_scaling_consistency()
    test_batch_scaling() 