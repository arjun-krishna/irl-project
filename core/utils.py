from core.dataset import MiniWorldDataset
from core.demo import Demonstration
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