import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import torch.optim as optim
from agents.net import MLP, Encoder
from core.transforms import NormalizeTensor, RandomCrop, ToTensor
from core.dataset import MiniWorldDataset

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion() # interactive mode

MODEL_PATH = "bc.pt"

# params
batch_size = 64
learning_rate = 1e-3
n_epochs = 20
# ---

dataset = MiniWorldDataset(
    fname='hallway_agent_view.pickle', 
    root_dir='dataset', 
    transform=transforms.Compose([
        RandomCrop(50),
        ToTensor(),
        NormalizeTensor()
    ]))

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

encoder = Encoder()
mlp = MLP()

weights = [1.5, 1.5, 0.1, 10.0, 1.0, 1.0, 1.0, 1.0] # moving forward is quite likely
class_weights = torch.FloatTensor(weights)
cross_entropy = nn.CrossEntropyLoss(weight=class_weights)
params = list(encoder.parameters()) + list(mlp.parameters())
optimizer = optim.Adam(params, lr=learning_rate)

for epoch in range(n_epochs):
    running_loss = 0.
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data['obs'], data['action']
        optimizer.zero_grad()

        outputs = encoder(inputs)
        outputs = mlp(outputs)

        loss = cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 50 == 0: # every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % 
                (epoch+1, i+1, running_loss / 2000))
            running_loss = 0.

torch.save({
    'encoder_dict': encoder.state_dict(),
    'mlp_dict': mlp.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}, MODEL_PATH)

print('done')