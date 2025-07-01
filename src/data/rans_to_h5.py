import h5py
import pyvista as pv
import os
import tqdm
import numpy as np
import torch
import pickle
from src.data import scaling

def calculate_normals(surface_points):
    N = len(surface_points)

    # Initialize array for normals
    normals = np.zeros_like(surface_points)

    # Loop through interior points
    for i in range(1, N - 1):
        prev = surface_points[i - 1]
        next = surface_points[i + 1]

        # Approximate tangent by vector between neighbors
        tangent = next - prev
        tangent /= np.linalg.norm(tangent)

        # Rotate 90° to get normal
        normals[i] = [-tangent[1], tangent[0]]

    # For the endpoints, use forward and backward difference
    tangent_start = surface_points[1] - surface_points[0]
    tangent_start /= np.linalg.norm(tangent_start)
    normals[0] = [-tangent_start[1], tangent_start[0]]

    tangent_end = surface_points[-1] - surface_points[-2]
    tangent_end /= np.linalg.norm(tangent_end)
    normals[-1] = [-tangent_end[1], tangent_end[0]]
    
    return normals

def rans_to_h5(vtu_dir = "./dataset/rans_naca0012/incompressible/dataset", 
               output_h5_dir = "./dataset/rans_naca0012/incompressible/dataset/h5_files",
               bounds = (-1.5, 10, -1, 5, 0, 0),
               scaling_type=4, 
               scaler_name='standard',
               sorted_vtu_files=None,
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
    
    train_vtu_path = os.path.join(vtu_dir, 'train')
    val_vtu_path = os.path.join(vtu_dir, 'val')
    
    train_vtu_files = sorted([f for f in os.listdir(train_vtu_path) if f.lower().endswith('.vtu')])
    val_vtu_files = sorted([f for f in os.listdir(val_vtu_path) if f.lower().endswith('.vtu')])
    
    #initialize dictionaries
    train_data_dict = {}
    val_data_dict = {}
    traj_num = 0

    for train_vtu_file in tqdm.tqdm(train_vtu_files, desc="Processing VTU files"):
        
        train_grid = pv.UnstructuredGrid(os.path.join(train_vtu_path, train_vtu_file))
        
        if bounds is not None:
            train_grid = train_grid.clip_box(bounds=bounds, invert=False, crinkle=True)
            
        """
        We need to extract the following data:
        - coordinates (x, y)
        - velocities (x, y)
        - edge_index
        - pressure
        - density
        - laminar viscosity
        - params (Re, alpha)
        - surface:
            - mask (bool)
            - normals (x, y)
            - cp
            - cf (x, y, z)
        """
        
        #name : flow_Re_{Re}_alpha_{alpha}.vtu
        train_Re = train_vtu_file.split('_')[2].split('.')[0]
        train_alpha = train_vtu_file.split('_')[4].split('.')[0]
        train_params = np.array([train_Re, train_alpha], dtype=np.float32)

        # volumevariables
        train_coordinates = np.array(train_grid.points[:, 0:2], dtype=np.float32)
        train_velocities = np.array(train_grid['Velocity'][:, 0:2], dtype=np.float32)
        train_edge_index = train_grid.extract_all_edges().lines.reshape(-1, 3)[:,1:].reshape(2, -1)
        train_pressure = np.array(train_grid['Pressure'], dtype=np.float32)
        train_density = train_grid['Density']
        train_laminar_viscosity = train_grid['Laminar_Viscosity']

        #surface variables
        train_surface_mask = ~torch.all(torch.tensor(train_grid['Skin_Friction_Coefficient']) == 0, dim=1)
        train_surface_points = train_coordinates[train_surface_mask]
        train_surface_normals = calculate_normals(train_surface_points)
        train_surface_cp = train_grid['Pressure_Coefficient'][train_surface_mask]
        train_surface_cf = train_grid['Skin_Friction_Coefficient'][train_surface_mask]
        
        #append to dictionary
        train_data_dict[f'configuration_{traj_num}'] = {
            'coordinates': train_coordinates,
            'Ux': train_velocities[:,0],
            'Uy': train_velocities[:,1],
            'edge_index': train_edge_index,
            'pressure': train_pressure,
            'density': train_density,
            'laminar_viscosity': train_laminar_viscosity,
            'params': train_params,
            'surface_mask': train_surface_mask,
            'surface_normals': train_surface_normals,
            'surface_cp': train_surface_cp,
            'surface_cf': train_surface_cf,
        }

        # Get corresponding val file if it exists, otherwise use None
        val_vtu_file = val_vtu_files[traj_num] if traj_num < len(val_vtu_files) else None

        if val_vtu_file is not None:
            val_grid = pv.UnstructuredGrid(os.path.join(val_vtu_path, val_vtu_file))
            if bounds is not None:
                val_grid = val_grid.clip_box(bounds=bounds, invert=False, crinkle=True)
                
            val_Re = val_vtu_file.split('_')[2].split('.')[0]
            val_alpha = val_vtu_file.split('_')[4].split('.')[0]
            val_params = np.array([val_Re, val_alpha], dtype=np.float32)

            # volumevariables
            val_coordinates = np.array(val_grid.points[:, 0:2], dtype=np.float32)
            val_velocities = np.array(val_grid['Velocity'][:, 0:2], dtype=np.float32)
            val_edge_index = val_grid.extract_all_edges().lines.reshape(-1, 3)[:,1:].reshape(2, -1)
            val_pressure = np.array(val_grid['Pressure'], dtype=np.float32)
            val_density = np.array(val_grid['Density'], dtype=np.float32)
            val_laminar_viscosity = val_grid['Laminar_Viscosity']

            val_surface_mask = ~torch.all(torch.tensor(val_grid['Skin_Friction_Coefficient']) == 0, dim=1)
            val_surface_points = val_coordinates[val_surface_mask]
            val_surface_normals = calculate_normals(val_surface_points)
            val_surface_cp = val_grid['Pressure_Coefficient'][val_surface_mask]
            val_surface_cf = val_grid['Skin_Friction_Coefficient'][val_surface_mask]
            
            val_data_dict[f'configuration_{traj_num}'] = {
                'coordinates': val_coordinates,
                'Ux': val_velocities[:,0],
                'Uy': val_velocities[:,1],
                'edge_index': val_edge_index,
                'pressure': val_pressure,
                'density': val_density,
                'laminar_viscosity': val_laminar_viscosity,
                'params': val_params,
                'surface_mask': val_surface_mask,
                'surface_normals': val_surface_normals,
                'surface_cp': val_surface_cp,
                'surface_cf': val_surface_cf,
            }

        traj_num += 1

    # ===== SCALING PROCEDURE =====
    print("Applying scaling to velocity data (Ux and Uy separately)...")
    
    # Collect all training Ux and Uy for scaling
    all_train_ux = []
    all_train_uy = []
    for config_key in train_data_dict.keys():
        ux = train_data_dict[config_key]['Ux']
        uy = train_data_dict[config_key]['Uy']
        all_train_ux.append(ux)
        all_train_uy.append(uy)
    
    # Stack all training velocities for scaling
    stacked_train_ux = np.concatenate(all_train_ux)  # Shape: [total_nodes]
    stacked_train_uy = np.concatenate(all_train_uy)  # Shape: [total_nodes]
    
    # Fit scalers on training data only (separate scalers for Ux and Uy)
    print(f"Fitting Ux scaler on training data of shape: {stacked_train_ux.shape}")
    ux_scaler, _ = scaling.tensor_scaling(torch.tensor(stacked_train_ux.reshape(-1, 1)), scaling_type, scaler_name)
    
    print(f"Fitting Uy scaler on training data of shape: {stacked_train_uy.shape}")
    uy_scaler, _ = scaling.tensor_scaling(torch.tensor(stacked_train_uy.reshape(-1, 1)), scaling_type, scaler_name)
    
    if ux_scaler is not None and uy_scaler is not None:
        print("Scalers fitted successfully. Applying to all data...")
        
        # Determine which scaler to use for transform (for scaling_type=4, use the first scaler)
        if isinstance(ux_scaler, list):
            ux_transform_scaler = ux_scaler[0]  # Use first scaler for transform
        else:
            ux_transform_scaler = ux_scaler
            
        if isinstance(uy_scaler, list):
            uy_transform_scaler = uy_scaler[0]  # Use first scaler for transform
        else:
            uy_transform_scaler = uy_scaler
        
        # Apply scaling to all configurations
        for config_key in train_data_dict.keys():
            original_ux = train_data_dict[config_key]['Ux']
            original_uy = train_data_dict[config_key]['Uy']
            
            scaled_ux = np.array(ux_transform_scaler.transform(original_ux.reshape(-1, 1))).flatten()
            scaled_uy = np.array(uy_transform_scaler.transform(original_uy.reshape(-1, 1))).flatten()
            
            train_data_dict[config_key]['Ux'] = scaled_ux
            train_data_dict[config_key]['Uy'] = scaled_uy
        
        for config_key in val_data_dict.keys():
            original_ux = val_data_dict[config_key]['Ux']
            original_uy = val_data_dict[config_key]['Uy']
            
            scaled_ux = np.array(ux_transform_scaler.transform(original_ux.reshape(-1, 1))).flatten()
            scaled_uy = np.array(uy_transform_scaler.transform(original_uy.reshape(-1, 1))).flatten()
            
            val_data_dict[config_key]['Ux'] = scaled_ux
            val_data_dict[config_key]['Uy'] = scaled_uy
        
        # Save the fitted scalers
        scalers = {'Ux_scaler': ux_scaler, 'Uy_scaler': uy_scaler}
        with open(scaler_path, 'wb') as f:
            pickle.dump(scalers, f)
        print(f"Scalers saved to {scaler_path}")
    else:
        print("Warning: Scaler fitting failed. Data will be saved without scaling.")
    
    # ===== WRITE TO H5 FILES =====
    print("Writing data to H5 files...")
    
    # Write training data
    with h5py.File(train_h5_path, 'w') as f:
        for config_key, config_data in train_data_dict.items():
            g = f.create_group(config_key)
            for key, value in config_data.items():
                g[key] = value
    
    # Write validation data
    with h5py.File(val_h5_path, 'w') as f:
        for config_key, config_data in val_data_dict.items():
            g = f.create_group(config_key)
            for key, value in config_data.items():
                g[key] = value
    
    print(f"Data successfully written to {train_h5_path} and {val_h5_path}")
    return train_h5_path, val_h5_path, scaler_path
