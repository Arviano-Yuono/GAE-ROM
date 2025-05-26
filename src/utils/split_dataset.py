import os
import random
import shutil
import tqdm

def split_dataset(data_dir,
                  split_dir, 
                  train_ratio=0.8, 
                  val_ratio=0.1, 
                  shuffle=True,
                  seed=42,
                  sample_size=None,
                  test_split = True,
                  overwrite=False):
    """
    Split the dataset into training, validation, and test sets.
    """
    if os.path.exists(split_dir) and not overwrite:
        print(f"split directory {split_dir} already exists, taking the existing split. Set overwrite to True to overwrite.")
        train_dir = os.path.join(split_dir, 'train')
        val_dir = os.path.join(split_dir, 'val')
        test_dir = os.path.join(split_dir, 'test')
        return train_dir, val_dir, test_dir
    elif os.path.exists(split_dir) and overwrite:
        shutil.rmtree(split_dir)
        os.makedirs(split_dir, exist_ok=True)

    list_files = []
    train_files = []
    val_files = []
    test_files = []

    for file in os.listdir(data_dir):
        if file.endswith('.vtu') or file.endswith('.vtk'):
            list_files.append(file)
        else:
            print(f"file type is not supported: {file}\n.vtu or .vtk is expected")

    if shuffle:
        random.seed(seed)
        random.shuffle(list_files)

    if sample_size:
        list_files = list_files[:sample_size]

    train_num = int(len(list_files) * train_ratio)
    val_num = int(len(list_files) * val_ratio)

    train_files = list_files[:train_num]
    val_files = list_files[train_num:train_num+val_num]
    if test_split:
        test_files = list_files[train_num+val_num:]
    else:
        test_files = []

    train_dir = os.path.join(split_dir, 'train')
    val_dir = os.path.join(split_dir, 'val')
    test_dir = os.path.join(split_dir, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    loop = tqdm.tqdm(train_files)
    for file in loop:
        shutil.copy(os.path.join(data_dir, file), os.path.join(train_dir, file))
    loop = tqdm.tqdm(val_files)
    for file in loop:
        shutil.copy(os.path.join(data_dir, file), os.path.join(val_dir, file))
    loop = tqdm.tqdm(test_files)
    for file in loop:
        shutil.copy(os.path.join(data_dir, file), os.path.join(test_dir, file))

    return train_dir, val_dir, test_dir


