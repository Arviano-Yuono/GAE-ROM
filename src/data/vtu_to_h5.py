import h5py
import pyvista as pv
import os
import tqdm
import numpy as np

def vtu_to_h5(vtu_file_directory, h5_file_path, overwrite=False, edge_index=None):
    if os.path.exists(h5_file_path) and not overwrite:
        print(f"File {h5_file_path} already exists. Set overwrite=True to overwrite.")
        return
    vtu_files = os.listdir(vtu_file_directory)
    with h5py.File(h5_file_path, 'w') as f:
        # Read first file to get coordinates
        first_grid = pv.read(os.path.join(vtu_file_directory, vtu_files[0]))
        coordinates = np.array(first_grid.points[:,0:2])
        # Store coordinates and edge_index at root level
        f['coordinates'] = coordinates
        f['edge_index'] = edge_index
        
        for vtu_file in tqdm.tqdm(vtu_files):
            grid = pv.read(os.path.join(vtu_file_directory, vtu_file))
            # vel data
            velocities = np.array(grid.point_data['Velocity'][:,:2])
            # Create group and store velocities
            g = f.create_group(str(vtu_file.split('.')[0]))
            g['Ux'] = velocities[:,0]
            g['Uy'] = velocities[:,1]
        print(f"Data successfully converted to {h5_file_path}")
