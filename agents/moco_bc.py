import torch
import argparse
import random
import os
from pathlib import Path
import gym
import gym_miniworld
import numpy as np
from torch import nn
import torch.optim as optim
from torchvision.transforms.transforms import RandomApply
from core.transforms import GaussianBlur, TwoCropsTransform
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from agents.net import MLP, Encoder
from PIL import Image

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default='MiniWorld-Hallway-v0', type=str, help='env name to evaluate on')
parser.add_argument('--top_view', action='store_true', help='show the top view instead of the agent view')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--data_path', default='', type=str, help='image dataset folder to train model from')
parser.add_argument('--batch_size', default=64, type=int, help='training batch size')
parser.add_argument('--nb_epochs', default=20, type=int, help='number of epochs')
parser.add_argument('--eval_epoch', default=10, type=int, help='run evaluate after specified epochs')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--seed', default=102, type=int, help='random seed')
args = parser.parse_args()

if not os.path.exists('models/'):
    os.makedirs('models/')
MODEL_PATH = 'models/' + Path(args.data_path).parts[-1] + '_moco_bc.pt'
MODEL_METRICS_PATH = 'models/' + Path(args.data_path).parts[-1] + '_moco_bc.pickle'

random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# moco params
Q = 640
m = 0.999
T = 0.04
# ---

encoder_q = Encoder()
encoder_k = Encoder()
mlp = MLP()

for param_q, param_k in zip(encoder_q.parameters(), encoder_k.parameters()):
    param_k.data.copy_(param_q.data)  # initialize
    param_k.requires_grad = False  # not update by gradient

weights = [1., 1., 0.1, 0.01, 0.01, 0.01, 0.01, 0.01]
class_weights = torch.FloatTensor(weights)
cross_entropy_task = nn.CrossEntropyLoss(weight=class_weights)

def moco_loss(q, k, queue):
    N = q.shape[0]
    C = q.shape[1]

    pos = torch.exp(torch.div(torch.bmm(q.view(N, 1, C), k.view(N, C, 1)).view(N, 1), T))
    neg = torch.sum(torch.exp(torch.div(torch.mm(q.view(N, C), torch.t(queue)), T)), dim=1)
    denom = neg + pos
    return torch.mean(-torch.log(torch.div(pos, denom)))

params = list(encoder_q.parameters()) + list(mlp.parameters())
optimizer = optim.Adam(params, lr=args.lr)

normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

augmentation = [
    transforms.RandomResizedCrop(50, scale=(0.2, 1.)),
    RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur([.1, .2])], p=0.5), 
    # do not apply horizontal flip (action could change based on that - not label preserving)
    transforms.ToTensor(),
    normalizer
]

train_dataset = datasets.ImageFolder(
    args.data_path,
    TwoCropsTransform(transforms.Compose(augmentation)))

train_loader = DataLoader(
    train_dataset, args.batch_size, shuffle=True, pin_memory=True, drop_last=True)

def evaluateInEnv():
    encoder_q.eval()
    mlp.eval()
    env = gym.make(args.env_name)
    env.reset()
    get_obs = env.render_top_view if args.top_view else env.render_obs

    if args.domain_rand:
        env.domain_rand = True

    NUM_EPISODES = 100

    metric_steps = []
    metric_success = []

    trsf = transforms.Compose([
        transforms.RandomResizedCrop(50, scale=(0.2, 1.)),
        transforms.ToTensor(),
        normalizer
    ])


    for i in range(NUM_EPISODES):
        done = False
        steps = 0
        while not done:
            x = Image.fromarray(get_obs())
            inp = trsf(x)
            output = encoder_q(inp[np.newaxis, :, :, :])
            output = mlp(output)
            output = output.clone().detach()
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
    encoder_q.train()
    mlp.train()
    return d

queue = None
# initialize queue
flag = False
while True:
    for _, (images, label) in enumerate(train_loader):
        xk = images[1]
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

encoder_q.train()
mlp.train()

metric_loss = []

for epoch in range(args.nb_epochs):
    # adjust_learning_rate(optimizer, epoch)

    epoch_loss = {
        'bc': [],
        'moco': [],
        'loss': []
    }
    
    for i, (images, labels) in enumerate(train_loader):

        optimizer.zero_grad()

        xq, xk = images[0], images[1]

        q = encoder_q(xq)
        k = encoder_k(xk).detach()

        q = torch.div(q, torch.norm(q, dim=1).reshape(-1, 1))
        k = torch.div(k, torch.norm(k, dim=1).reshape(-1, 1))

        outputs = mlp(q)
        
        l_moco = moco_loss(q, k, queue)
        l_bc = cross_entropy_task(outputs, labels)

        loss = l_moco + l_bc

        epoch_loss['moco'].append(l_moco.item())
        epoch_loss['bc'].append(l_bc.item())
        epoch_loss['loss'].append(loss.item())
        
        loss.backward()
        optimizer.step()

        queue = torch.cat((queue, k), 0)
        if queue.shape[0] > Q:
            queue = queue[args.batch_size:,:]

        # momentum update key encoder
        for param_q, param_k in zip(encoder_q.parameters(), encoder_k.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)

    avg_bc_loss = np.mean(epoch_loss['bc'])
    avg_moco_loss = np.mean(epoch_loss['moco'])
    avg_loss = np.mean(epoch_loss['loss'])
    print(f"Epoch {epoch} | BC-Loss={avg_bc_loss} | Moco-Loss={avg_moco_loss} | Loss={avg_loss}")
    metric_loss.append(avg_loss)

torch.save({
    'encoder_dict': encoder_q.state_dict(),
    'mlp_dict': mlp.state_dict()
}, MODEL_PATH)


# plt.plot(range(1, args.nb_epochs + 1), metric_loss, '-bo')
# plt.title('Average Loss vs Epoch')
# plt.xlabel('epoch')
# plt.ylabel('Avg. Loss')
# plt.show()