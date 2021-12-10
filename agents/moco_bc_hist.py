"""
Action considered BC
"""
import argparse
import pickle
import torch
import random
import gym
import gym_miniworld
import os
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from core.transforms import GaussianBlur, TwoCropsTransform
from torchvision.transforms.transforms import RandomApply
import torch.optim as optim
from PIL import Image
from agents.net import MLP, Encoder
from agents.hist_dataset import DemoDataPreviousAction
from core.model_metrics import ModelMetrics

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default='MiniWorld-OneRoom-v0', type=str, help='env name to evaluate on')
parser.add_argument('--top-view', action='store_true', help='show the top view instead of the agent view')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--data-path', default='', type=str, help='folder of demos')
parser.add_argument('--nb-demos', default=50, type=int, help='number of demos to consider for training')
parser.add_argument('--batch-size', default=64, type=int, help='training batch size')
parser.add_argument('--nb-epochs', default=20, type=int, help='number of epochs')
parser.add_argument('--eval-epoch', default=10, type=int, help='run evaluate after specified epochs')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--seed', default=101, type=int, help='random seed')
parser.add_argument('--no-hist', action='store_true', help='remove history action feature')
args = parser.parse_args()

if not os.path.exists('models/'):
    os.makedirs('models/')
MODEL_NAME = 'D' + str(args.nb_demos) + '_moco_bc' + ('_hist' if not args.no_hist else '')
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

augmentation = [
    transforms.RandomResizedCrop(64, scale=(0.2, 1.)),
    RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur([.1, .2])], p=0.5), 
    # do not apply horizontal flip (action could change based on that - not label preserving)
    transforms.ToTensor(),
    normalizer
]

data_transform = TwoCropsTransform(transforms.Compose(augmentation))

device = get_default_device()
print(f'Using device = {device}')

train_dataset = DemoDataPreviousAction(args.data_path, args.nb_demos, data_transform)
train_loader = DeviceDataLoader(DataLoader(train_dataset, args.batch_size, shuffle=True), device)

# moco params
Q = 128#640
m = 0.999
T = 0.04
# ---

encoder_q = to_device(Encoder(), device)
encoder_k = to_device(Encoder(), device)
out_dim = 4096
if args.no_hist:
    mlp = to_device(MLP(dim=out_dim), device)
else:
    mlp = to_device(MLP(dim=out_dim + 9), device)

for param_q, param_k in zip(encoder_q.parameters(), encoder_k.parameters()):
    param_k.data.copy_(param_q.data)  # initialize
    param_k.requires_grad = False  # not update by gradient

def model_forward(sample):
    e = encoder_q(sample['obs'][0])
    if args.no_hist:
        t = e
    else:
        t = torch.concat([e, sample['prev_a']], dim=1)
    return mlp(t), e

def evaluateInEnv():
    env = gym.make(args.env_name)
    env.reset()
    get_obs = env.render_top_view if args.top_view else env.render_obs

    if args.domain_rand:
        env.domain_rand = True

    NUM_EPISODES = 50

    metric_steps = []
    metric_success = []

    trsf = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.2, 1.)),
        transforms.ToTensor(),
        normalizer
    ])

    for i in range(NUM_EPISODES):
        done = False
        steps = 0
        prev_a = 8
        while not done:
            sample = {
                'obs': [torch.unsqueeze(trsf(Image.fromarray(get_obs())), 0)],
                'prev_a': F.one_hot(torch.tensor([prev_a]), num_classes=9)
            }
            output, _ = model_forward(to_device(sample, device))
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

def moco_loss(q, k, queue):
    N = q.shape[0]
    C = q.shape[1]

    pos = torch.exp(torch.div(torch.bmm(q.view(N, 1, C), k.view(N, C, 1)).view(N, 1), T))
    neg = torch.sum(torch.exp(torch.div(torch.mm(q.view(N, C), torch.t(queue)), T)), dim=1)
    denom = neg + pos
    return torch.mean(-torch.log(torch.div(pos, denom)))

weights = [1., 1., 0.4, 0.01, 0.01, 0.01, 0.01, 0.01] # moving forward is quite likely
class_weights = to_device(torch.FloatTensor(weights), device)
cross_entropy = nn.CrossEntropyLoss(weight=class_weights)
#cross_entropy = nn.CrossEntropyLoss()
params = list(encoder_q.parameters()) + list(mlp.parameters())
optimizer = optim.Adam(params, lr=args.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.nb_epochs)

model_metrics = ModelMetrics(MODEL_NAME)

if device.type == 'cuda':
    torch.cuda.empty_cache()

queue = None
# initialize queue
flag = False
while True:
    for _, sample in enumerate(train_loader):
        xk = sample['obs'][1]
        k = encoder_k(xk).detach()

        k = torch.div(k, torch.norm(k, dim=1).reshape(-1, 1))

        if queue is None:
            queue = k
        else:
            if queue.shape[0] < Q:
                queue = torch.cat((queue, k), 0)
            else:
                flag = True
        if flag:
            break
    if flag:
        break
#end queue init

encoder_q.train(); mlp.train()
for epoch in range(args.nb_epochs):
    epoch_loss = {
        'bc': [],
        'moco': [],
        'loss': []
    }
    for i, sample in enumerate(train_loader):
        optimizer.zero_grad()
        a_pred, q = model_forward(sample)
        k = encoder_k(sample['obs'][1]).detach()
        loss_bc = cross_entropy(a_pred, sample['a'].squeeze())

        q = torch.div(q, torch.norm(q, dim=1).reshape(-1, 1))
        k = torch.div(k, torch.norm(k, dim=1).reshape(-1, 1))
        loss_moco = moco_loss(q, k, queue)

        loss = loss_bc + loss_moco
        loss.backward()
        optimizer.step()

        epoch_loss['moco'].append(loss_moco.cpu().item())
        epoch_loss['bc'].append(loss_bc.cpu().item())
        epoch_loss['loss'].append(loss.cpu().item())

        queue = torch.cat((queue, k), 0)
        if queue.shape[0] > Q:
            queue = queue[args.batch_size:,:]

        # momentum update key encoder
        for param_q, param_k in zip(encoder_q.parameters(), encoder_k.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)

    scheduler.step()
    avg_bc_loss = np.mean(epoch_loss['bc'])
    avg_moco_loss = np.mean(epoch_loss['moco'])
    avg_loss = np.mean(epoch_loss['loss'])
    print(f"Epoch {epoch} | BC-Loss={avg_bc_loss} | Moco-Loss={avg_moco_loss} | Loss={avg_loss}")
    
    model_metrics.add_loss(avg_loss)
    model_metrics.add_epoch_metric(epoch, {
        'bc_loss': avg_bc_loss,
        'moco_loss': avg_moco_loss
    })
    
    if (epoch+1) % args.eval_epoch == 0:
        encoder_q.eval(); mlp.eval()
        eval_result = evaluateInEnv()
        print('EVAL (success_rate) = ', eval_result['success_rate'])
        model_metrics.add_eval(epoch, eval_result) 
        encoder_q.train(); mlp.train()

encoder_q.eval(); mlp.eval()
eval_result = evaluateInEnv()
model_metrics.add_eval(-1, eval_result) 

torch.save({
    'encoder_dict': encoder_q.state_dict(),
    'mlp_dict': mlp.state_dict()
}, MODEL_PATH)

with open(MODEL_METRICS_PATH, 'wb') as fp:
    pickle.dump(model_metrics.getDict(), fp)