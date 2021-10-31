from core.demo import Demonstration
import pickle
import os
import time

def store_demo(demo: Demonstration, env_name: str, view_type: str, folder: str = 'demos') -> None:
    """Store serialized demonstration as folder/env_name/view_type/demo_{timestamp}
    """
    file_name = 'demo_' + str(int(time.time())) + '.pickle'
    fpath = os.path.join(folder, env_name, view_type)
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
