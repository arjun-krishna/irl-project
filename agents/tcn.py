import torch
import argparse
import random
from torch import nn
from torch.nn import functional as F
import os
from pathlib import Path
import numpy as np
import torch.optim as optim
from torchvision.transforms.transforms import RandomApply
from core.transforms import GaussianBlur, TwoCropsTransform
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from agents.net import MLP, Encoder

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
plt.style.use('ggplot')

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='', type=str, help='image dataset folder to train model from')
parser.add_argument('--batch_size', default=64, type=int, help='training batch size')
parser.add_argument('--nb_epochs', default=20, type=int, help='number of epochs')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--margin', type=float, default=2.0, help='learning rate')
parser.add_argument('--seed', default=102, type=int, help='random seed')
args = parser.parse_args()

if not os.path.exists('models/'):
    os.makedirs('models/')
MODEL_PATH = 'models/' + Path(args.data_path).parts[-1] + 'tcn.pt'

random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# TODO: Continue from below...

def triplet_loss(margin, d_positive, d_negative):
    loss = torch.clamp(margin + d_positive - d_negative, min=0.0).mean()
    return loss

def distance(x1, x2):
    diff = torch.abs(x1 - x2)
    return torch.pow(diff, 2).sum(dim=1)


train_dataset = datasets.ImageFolder(args.data_path)

train_loader = DataLoader(
        train_dataset,
        args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True)
