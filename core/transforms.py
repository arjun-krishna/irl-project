import torch
import numpy as np

class RandomCrop(object):
    """Crop randomly the image in a sample
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
    
    def __call__(self, sample):
        obs, action = sample['obs'], sample['action']

        h, w = obs.shape[:2]
        new_h, new_w = self.output_size

        top =  np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        obs = obs[top: top + new_h,
                  left: left + new_w]
        
        return {
            'obs': obs,
            'action': action
        }


class ToTensor(object):
    """Convert ndarrays in sample to Tensors"""

    def __call__(self, sample):
        obs, action = sample['obs'], sample['action']

        # swap color axis 
        # numpy image: H x W x C
        # torch image: C x H x W
        obs = obs.transpose((2, 0, 1))
        return {
            'obs': torch.from_numpy(obs),
            'action': torch.tensor(action)
        }

class NormalizeTensor(object):
    """Normalize tensor between -1 and 1"""

    def __call__(self, sample):
        obs, action = sample['obs'], sample['action']
        obs = (obs - 127.5) / 127.5
        return {
            'obs': obs,
            'action': action
        }