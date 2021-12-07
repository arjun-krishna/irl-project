import torch
import torch.nn as nn
import torch.optim as optim
from agents.net import MLP, Encoder

from torchvision import transforms as T

class ContextPredictionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.mlp_cp = MLP(dim=3200)
        self.mlp_bc = MLP(dim=150553)

    def forward(self, x):
        patch1, patch2 = x
        features1 = self.encoder(patch1)
        features2 = self.encoder(patch2)
        features = torch.cat((features1, features2))
        out = self.mlp(features)
        return out

    
