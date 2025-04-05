import h5py
import pyvista as pv
import os
import tqdm
import numpy as np

def vtu_to_h5(vtu_file_directory, h5_file_path):
    vtu_files = os.listdir(vtu_file_directory)
    with h5py.File(h5_file_path, 'w') as f:
        for vtu_file in tqdm.tqdm(vtu_files):
            grid = pv.read(os.path.join(vtu_file_directory, vtu_file))

            # coordinates
            coordinates = np.array(grid.points[:,0:2])

            # vel data
            velocities = np.array(grid.point_data['Velocity'][:,:2])
            # data_dict = {
            #     "coordinates": {"x": coordinates[:,0], "y": coordinates[:,1]},
            #     "velocity": {"x": velocities[:,0], "y": velocities[:,1]}
            # }
            g = f.create_group(str(vtu_file.split('.')[0])) 
            g['coordinates'] = coordinates
            g['velocity'] = velocities

        print(f"Data successfully converted to {h5_file_path}")



