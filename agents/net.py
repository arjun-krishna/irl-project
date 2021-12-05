import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # In: 3x50x50
        self.conv1 = nn.Conv2d(3, 6, 7) # Out:   6x44x44
        self.pool = nn.MaxPool2d(2, 2)  # Out:   6x22x22
        self.conv2 = nn.Conv2d(6, 10, 7) # Out: 10x16x16
        self.conv3 = nn.Conv2d(10, 16, 7) # Out: 16x10x10
        # self.fc1 = nn.Linear(16 * 10 * 10, 256)
        # self.fc2 = nn.Linear(256, dim)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        return x

class MLP(nn.Module):
    def __init__(self, dim=128) -> None:
        super().__init__()
        # In: 120
        self.fc1 = nn.Linear(dim, 64) # Out: 64
        self.fc2 = nn.Linear(64, 32)  # Out: 32
        self.fc3 = nn.Linear(32, 8)  # Out: 8
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x