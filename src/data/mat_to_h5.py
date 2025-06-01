import os
import pickle
import pandas as pd
import scipy.io
import h5py
import numpy as np
import torch
import tqdm
# import scaling
from src.data import scaling

def mat_to_h5(mat_file_path, 
              h5_file_path, 
              variables = ['VX'],
              train_ratio = 0.1, 
              val_ratio = 0.9, 
              scaling_type = 4, 
              scaler_name = 'standard', 
              overwrite=False):
    
    train_path = os.path.join(h5_file_path, 'train.h5')
    val_path = os.path.join(h5_file_path, 'val.h5')
    
    if os.path.exists(train_path) and not overwrite:
        print(f"File {train_path} already exists. Set overwrite=True to overwrite.")
        return
    if not os.path.exists(h5_file_path):
        os.makedirs(h5_file_path)
    
    data_mat = scipy.io.loadmat(mat_file_path)
    train_trajectories, val_trajectories = split_dataset_trajectories(data_mat['VX'], train_ratio, val_ratio)
    
    # scaling
    if len(variables) == 1:
        print('SINGLE VARIABLE')
        all_velocities = torch.tensor(data_mat[variables[0]])
        val_velocities = all_velocities[:, val_trajectories]
        
        scaler_all, scaled_velocities   = scaling.tensor_scaling(all_velocities, scaling_type, scaler_name) # shape: (num_grids, num_nodes)
        print(scaled_velocities[0,:,:])
        scaler_test, scaled_val_velocities = scaling.tensor_scaling(val_velocities, scaling_type, scaler_name) # shape: (num_grids, num_nodes)
    else:
        print('MULTIPLE VARIABLES')
        var1_velocities = torch.tensor(data_mat[variables[0]])
        var2_velocities = torch.tensor(data_mat[variables[1]])
        all_velocities = torch.stack((var1_velocities, var2_velocities), dim=2)
        val_velocities = all_velocities[:, val_trajectories]
        
        var1_test = var1_velocities[:, val_trajectories]
        var2_test = var2_velocities[:, val_trajectories]
        scaler_var1_all, VAR1_all = scaling.tensor_scaling(var1_velocities, scaling_type, scaler_name)
        scaler_var1_test, VAR1_test = scaling.tensor_scaling(var1_test, scaling_type, scaler_name)
        scaler_var2_all, VAR2_all = scaling.tensor_scaling(var2_velocities, scaling_type, scaler_name)
        scaler_var2_test, VAR2_test = scaling.tensor_scaling(var2_test, scaling_type, scaler_name)
        scaled_velocities = torch.stack((VAR1_all, VAR2_all), dim=2)
        scaled_val_velocities = torch.stack((VAR1_test, VAR2_test), dim=2)
        scaler_all = [scaler_var1_all, scaler_var2_all]
        scaler_test = [scaler_var1_test, scaler_var2_test]

    # save scaler
    with open(os.path.join(h5_file_path, 'scaler_all.pkl'), 'wb') as f:
        pickle.dump(scaler_all, f)
    with open(os.path.join(h5_file_path, 'scaler_test.pkl'), 'wb') as f:
        pickle.dump(scaler_test, f)

    print(f"Train trajectories: {train_trajectories}")
    print(f"Val trajectories: {val_trajectories}")

    print(f"scaled_velocities shape before writing: {scaled_velocities.shape}")
    write_h5_file(data_mat= data_mat, velocities= scaled_velocities, trajectories= train_trajectories, h5_file_path= train_path)
    write_h5_file(data_mat= data_mat, velocities= scaled_velocities, trajectories= val_trajectories, h5_file_path= val_path)
    return scaled_velocities,train_trajectories, val_trajectories, scaler_all, scaler_test

def split_dataset_trajectories(all_trajectories:pd.DataFrame, train_ratio, val_ratio):
    total_sims = all_trajectories.shape[1]
    train_sims = int(train_ratio * total_sims)
    main_loop = list(range(total_sims))
    np.random.shuffle(main_loop)
    train_trajectories = main_loop[0:train_sims]
    train_trajectories.sort()
    val_trajectories = main_loop[train_sims:total_sims]
    val_trajectories.sort()
    return train_trajectories, val_trajectories

def write_h5_file(data_mat, velocities, trajectories:list[int], h5_file_path, variables = ['VX']):
    with h5py.File(h5_file_path, 'w') as f:
        # Store coordinates at root level if they exist
        for trajectory in trajectories:
            g = f.create_group(f'configuration_{trajectory}')
            if 'coordinates' in data_mat.keys():
                g['coordinates'] = data_mat['coordinates'][:,trajectory]
            elif 'xx' in data_mat.keys() and 'yy' in data_mat.keys():
                x_coords = data_mat['xx'][:,trajectory].flatten()
                y_coords = data_mat['yy'][:,trajectory].flatten()
                coords = np.column_stack((x_coords, y_coords))
                g['coordinates'] = coords
            
            g['edge_index'] = data_mat['E'].reshape(2,-1) - 1 # since the edge index is 1-indexed in the mat file
            
            if len(variables) == 1:
                g['Ux'] = velocities[trajectory,:]
            else:
                g['Ux'] = velocities[trajectory,:,0]
                g['Uy'] = velocities[trajectory,:,1]   
        print(f"Data successfully converted to {h5_file_path}")