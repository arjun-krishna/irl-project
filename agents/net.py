import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # In: 3x64x64# In2: 3x256x256
        self.blk1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1),  
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)          # 64x32x32 | 64x128x128
        )
        self.blk2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2)          # 128x16x16 | 128x64x64
        )
        self.blk3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)          # 64x8x8  | 64x32x32
        )  
    
    def forward(self, x):
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = torch.flatten(x, 1)
        return x

class MLP(nn.Module):
    def __init__(self, dim=128, p=0.4) -> None:
        super().__init__()
        # In: 120
        self.fc = nn.Sequential(
            nn.Linear(dim, 64), # Out: 64
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
            nn.Linear(64, 32),  # Out: 32
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
            nn.Linear(32, 8)
        )
    
    def forward(self, x):
        return self.fc(x)

##############################################
#
#   Classes related to SimSiam
#
##############################################

class SimSiam(nn.Module):
    def __init__(self, encoder=None, predictor=None, dim=2304, pred_dim=64):
        super(SimSiam, self).__init__()
        self.dim=dim
        if encoder is None:
            self.encoder = Encoder()
        else:
            self.encoder = encoder
        if predictor is None:
            # self.predictor = MLP(dim)
            self.predictor = nn.Sequential(
                nn.Linear(dim, pred_dim, bias=False),
                # nn.BatchNorm1d(pred_dim),
                nn.ReLU(inplace=True), # hidden layer
                nn.Linear(pred_dim, dim)
            )
        else:
            self.predictor = predictor
        
    def forward(self, x1, x2):
        z1 = self.encoder(x1) # NxC
        z2 = self.encoder(x2) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        return p1, p2, z1.detach(), z2.detach()
