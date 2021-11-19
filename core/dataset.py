import numpy as np
import torch
from torch.utils.data import Dataset

import pickle
import os

class MiniWorldDataset(Dataset):

    observations = np.array([])
    actions = np.array([])

    def __init__(self, fname: str = None, root_dir: str = 'dataset', transform=None):
        """
        Args:
            fname (string): Name of the datafile,
            dir (string): Path of 
        """
        self.transform = transform
        data_path = os.path.join(root_dir, fname)
        with open(data_path, 'rb') as fp:
            self.setDict(pickle.load(fp))
        print(f'Loaded {self.__len__()} records from {data_path}')
    
    def setDict(self, data_dict: dict):
        for field in ['obs', 'action']:
            assert field in data_dict
        
        self.observations = np.array(data_dict['obs'])
        # a = np.array(data_dict['action'])
        self.actions = np.array(data_dict['action'], dtype=np.int64)
        #np.zeros((a.size, 8)) # 8 actions for one-hot encoding
        #self.actions[np.arange(a.size), a] = 1.        

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {
            'obs' : self.observations[idx],
            'action': self.actions[idx]
        }
        if self.transform:
            sample = self.transform(sample)
        return sample
        