from torch._C import dtype
from core.utils import load_demo
import numpy as np
from numpy import random
from numpy.core.defchararray import center
import torch
from torch.utils.data import Dataset
from core.hist_dataset import DemoDataPreviousAction
import pickle
import os
from torchvision import transforms as T
from PIL import Image

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
        
class ContextPredictionDataset(DemoDataPreviousAction):
    def __init__(self, demo_folder, input_shape='auto_infer'):
        super().__init__(demo_folder=demo_folder)
        self.input_shape = input_shape
        if self.input_shape == 'auto_infer':
            self.input_shape = self.get_input_shape_from_demo_folder(demo_folder)
            self.transform = T.Compose([T.ToTensor()])
        else:
            self.transform = T.Compose([T.Resize(input_shape), T.ToTensor()])
        
        self.center = np.array(self.input_shape)//2
        # print('Input Shape:', self.input_shape)
        self.patch_size = (min(self.input_shape))//4.5
        # print('Patch Size:', self.patch_size)
        r = 3*self.patch_size/2
        # print('r:', r)
        thetas = (np.pi/4)*np.arange(8)
        # print('thetas:', thetas)
        self.label2vec = r*np.stack([np.cos(thetas), np.sin(thetas)])
        print(type(self.label2vec))

    def __getitem__(self, idx):
        img = self.obs[idx]
        a = torch.from_numpy(self.a[idx])
        img = Image.fromarray(img)
        img = self.transform(img)
        # print('Image shape: ', img.shape)
        # print('Image type:     ', type(img))
        label = torch.randint(low=0, high=8, size=[1])
        # print('patch size:', self.patch_size)
        jitter1 = np.random.randint(low=1, high=7*self.patch_size//48)
        if np.random.rand() < 0.5:
            jitter1 *= -1
        jitter2 = jitter1
        if np.random.rand() < 0.5:
            jitter2 *= -1
        jitter = np.array([jitter1, jitter2])
        # print(jitter, type(jitter))
        # print(self.center, type(self.center))
        # print(self.label2vec[:,label], type(self.label2vec[:,label]))
        # print(self.center + self.label2vec[:,label] + jitter)
        random_loc = np.array(self.center + self.label2vec[:,label] + jitter, dtype=np.uint8)
        # print('Random Loc: ', random_loc)
        # print('Center: ', self.center)
        center_patch = self.get_patch(img, self.center)
        random_patch = self.get_patch(img, random_loc)
        # print('Center Patch Shape: ', center_patch.shape)
        # print('Random Patch Shape: ', random_patch.shape)
        x = {
            'center': center_patch,
            'random': random_patch,
            'obs': img,
            'prev_a': a.long()
        }
        # print('X Shape: ', x.shape)
        # print('Rs:', random_patch.shape)
        # print('Cs:', center_patch.shape)
        return (x, label)

    def get_input_shape_from_demo_folder(self, demo_folder):
        file_path = os.path.join(demo_folder, os.listdir(demo_folder)[0])
        d = load_demo(file_path)
        return np.array(d.observations[0].shape[:2])

    def get_patch(self, img, center):
        patch_shape = np.array([self.patch_size, self.patch_size])
        low = np.array(center - patch_shape//2 - 1, dtype=np.uint8)
        high = np.array(center + patch_shape//2 + 1, dtype=np.uint8)
        # print('low', low)
        # print('high', high)
        # print('Low, High', low, high)
        patch = img[:,low[0]:high[0], low[1]:high[1]]
        patch = patch.float()
        # print('Patch Shape:', patch.shape)
        # print('Patch type:', type(patch))
        return patch