from core.dataset import MiniWorldDataset
from core.demo import Demonstration
from PIL import Image
import numpy as np
import pickle
import os
import time
import glob

def store_demo(demo: Demonstration, env_name: str, view_mode: str, demos_folder: str = 'demos') -> None:
    """Store serialized demonstration as folder/env_name/view_type/demo_{timestamp}
    """
    file_name = 'demo_' + str(int(time.time())) + '.pickle'
    fpath = os.path.join(demos_folder, env_name, view_mode)
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    f = os.path.join(fpath, file_name)
    with open(f, 'wb') as fp:
        pickle.dump(demo.getDict(), fp)
    print(f'stored demonstration with {len(demo.observations)} state/action pairs as {f}')

def load_demo(file_name: str) -> Demonstration:
    d = Demonstration()
    with open(file_name, 'rb') as fp:
        data = pickle.load(fp)
        d.setDict(data)
    return d

def create_dataset(env_name: str, view_mode: str, dataset_fname: str, demos_folder: str = 'demos') -> None:
    dir_path = os.path.join(demos_folder, env_name, view_mode, '*')
    dataset = {
        'obs': [],
        'action': []
    }
    for demo_file in glob.glob(dir_path):
        d = load_demo(demo_file)
        dataset['obs'] += d.observations
        dataset['action'] += d.actions
    with open(dataset_fname, 'wb') as fp:
        pickle.dump(dataset, fp)
    print(f'stored dataset at {dataset_fname} with {len(dataset["obs"])} records')

def create_image_dataset(env_name: str, view_mode: str, dataset_root: str = 'dataset', demos_folder: str = 'demos', num_demos=250, seed=42) -> None:
    demos_dir_path = os.path.join(demos_folder, env_name, view_mode, '*')
    counter = 0
    all_demos = [demo_file for demo_file in glob.glob(demos_dir_path)]
    np.random.seed(seed)
    selected_demos = np.random.choice(all_demos, size=num_demos, replace=False)
    dataset_dir = os.path.join(dataset_root, env_name, view_mode, 'D'+str(num_demos))
    for demo_file in selected_demos:
        d = load_demo(demo_file)
        for (obs, a) in zip(d.observations, d.actions):
            im = Image.fromarray(obs)
            p = os.path.join(dataset_dir, str(a.value))
            if not os.path.exists(p):
                os.makedirs(p)
            im.save(os.path.join(p, str(counter) + '.png'))
            counter += 1
    print('Done writing image folder with labels')