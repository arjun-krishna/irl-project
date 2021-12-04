"""
Action considered BC
"""
import argparse
import pickle
from numpy.core.defchararray import mod
import torch
import random
import gym
import gym_miniworld
import os
from pathlib import Path
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from PIL import Image
from agents.net import MLP, Encoder
from agents.hist_dataset import DemoDataPreviousAction
from core.model_metrics import ModelMetrics

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default='MiniWorld-Hallway-v0', type=str, help='env name to evaluate on')
parser.add_argument('--top-view', action='store_true', help='show the top view instead of the agent view')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--data-path', default='', type=str, help='folder of demos')
parser.add_argument('--nb-demos', default=50, type=int, help='number of demos to consider for training')
parser.add_argument('--batch-size', default=64, type=int, help='training batch size')
parser.add_argument('--nb-epochs', default=20, type=int, help='number of epochs')
parser.add_argument('--eval-epoch', default=10, type=int, help='run evaluate after specified epochs')
parser.add_argument('--lr', default=5e-3, type=float, help='learning rate')
parser.add_argument('--seed', default=101, type=int, help='random seed')
args = parser.parse_args()

if not os.path.exists('models/'):
    os.makedirs('models/')
MODEL_NAME = 'D' + str(args.nb_demos) + '_bc_hist'
MODEL_PATH = 'models/' + MODEL_NAME + '.pt'
MODEL_METRICS_PATH = 'models/' + MODEL_NAME + '.pickle'

random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

### GPU data transfer utils
def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    if isinstance(data, dict):
        return dict((key, to_device(data[key], device)) for key in data)
    return data.to(device, non_blocking=True)

class DeviceDataLoader:
    """Transfer data to the device and yield the transfered data"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    
    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)
    
    def __len__(self):
        return len(self.dl)
## utils end

normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((50, 50)),
    normalizer
])

device = get_default_device()
print(f'Using device = {device}')

train_dataset = DemoDataPreviousAction(args.data_path, args.nb_demos, data_transform)
train_loader = DeviceDataLoader(DataLoader(train_dataset, args.batch_size, shuffle=True), device)

class ModelWithHist(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder(dim=128)
        self.mlp = MLP(dim=137)
    
    def forward(self, sample):
        e = self.enc(sample['obs'])
        X = torch.concat([e, sample['prev_a']], dim=1)
        return self.mlp(X)

model = to_device(ModelWithHist(), device)

def evaluateInEnv():
    env = gym.make(args.env_name)
    env.reset()
    get_obs = env.render_top_view if args.top_view else env.render_obs

    if args.domain_rand:
        env.domain_rand = True

    NUM_EPISODES = 100

    metric_steps = []
    metric_success = []

    for i in range(NUM_EPISODES):
        done = False
        steps = 0
        prev_a = 8
        while not done:
            sample = {
                'obs': data_transform(Image.fromarray(get_obs()))[np.newaxis, :, :, :],
                'prev_a': F.one_hot(torch.tensor([prev_a]), num_classes=9)
            }
            output = model(to_device(sample, device))
            output = output.clone().detach().cpu()
            action = output[0].argmax().item()

            _, reward, done, _ = env.step(action)
            steps += 1

            if done:
                env.reset()
                metric_success.append(reward > 0)
                metric_steps.append(steps)
    d = {
        'success_rate': np.mean(metric_success),
        'metric_steps': metric_steps
    }
    return d

weights = [1., 1., 0.4, 0.01, 0.01, 0.01, 0.01, 0.01] # moving forward is quite likely
class_weights = torch.FloatTensor(weights)
cross_entropy = nn.CrossEntropyLoss()
params = model.parameters()
optimizer = optim.Adam(params, lr=args.lr)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

model_metrics = ModelMetrics(MODEL_NAME)

if device.type == 'cuda':
    torch.cuda.empty_cache()
model.train()
for epoch in range(args.nb_epochs):
    epoch_loss = []
    for i, sample in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(sample)
        loss = cross_entropy(outputs, sample['a'].squeeze())
        loss.backward()
        epoch_loss.append(loss.detach().cpu().item())
        optimizer.step()
    scheduler.step()
    avg_loss = np.mean(epoch_loss)
    print('[%d] loss: %.3f' % (epoch, avg_loss))
    
    model_metrics.add_loss(avg_loss)

    if (epoch+1) % args.eval_epoch == 0:
        model.eval()
        eval_result = evaluateInEnv()
        print('EVAL (success_rate) = ', eval_result['success_rate'])
        model_metrics.add_eval(epoch, eval_result) 
        model.train()


model.eval()      
eval_result = evaluateInEnv()
model_metrics.add_eval(-1, eval_result) 

torch.save({
    'model_dict': model.state_dict()
}, MODEL_PATH)

with open(MODEL_METRICS_PATH, 'wb') as fp:
    pickle.dump(model_metrics.getDict(), fp)