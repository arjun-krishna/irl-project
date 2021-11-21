import argparse
import torch
import random
import os
from pathlib import Path
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.optim as optim
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
parser.add_argument('--seed', default=101, type=int, help='random seed')
args = parser.parse_args()

if not os.path.exists('models/'):
    os.makedirs('models/')
MODEL_PATH = 'models/' + Path(args.data_path).parts[-1] + '_bc.pt'

random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

train_dataset = datasets.ImageFolder(
    args.data_path,
    transforms.Compose([
        transforms.Resize((50, 50)),
        transforms.ToTensor(),
        normalizer
    ])
)

train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)

encoder = Encoder()
mlp = MLP()

weights = [1., 1., 0.1, 0.01, 0.01, 0.01, 0.01, 0.01] # moving forward is quite likely
class_weights = torch.FloatTensor(weights)
cross_entropy = nn.CrossEntropyLoss(weight=class_weights)
params = list(encoder.parameters()) + list(mlp.parameters())
optimizer = optim.Adam(params, lr=args.lr)


metric_loss = []

encoder.train()
mlp.train()

for epoch in range(args.nb_epochs):
    epoch_loss = []
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        outputs = encoder(inputs)
        outputs = mlp(outputs)

        loss = cross_entropy(outputs, labels)
        epoch_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    avg_loss = np.mean(epoch_loss)
    print('[%d] loss: %.3f' % (epoch, avg_loss))
    metric_loss.append(avg_loss)

torch.save({
    'encoder_dict': encoder.state_dict(),
    'mlp_dict': mlp.state_dict()
}, MODEL_PATH)

plt.plot(range(1, args.nb_epochs + 1), metric_loss, '-bo')
plt.title('Average BC Loss vs Epoch')
plt.xlabel('epoch')
plt.ylabel('Avg. Loss')
plt.show()
