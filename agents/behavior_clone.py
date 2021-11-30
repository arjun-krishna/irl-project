import argparse
import pickle
import torch
import random
import gym
import gym_miniworld
import os
from pathlib import Path
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.optim as optim
from PIL import Image
from agents.net import MLP, Encoder
from core.model_metrics import ModelMetrics

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
parser.add_argument('--seed', default=101, type=int, help='random seed')
args = parser.parse_args()

if not os.path.exists('models/'):
    os.makedirs('models/')
MODEL_NAME = Path(args.data_path).parts[-1] + '_bc'
MODEL_PATH = 'models/' + MODEL_NAME + '.pt'
MODEL_METRICS_PATH = 'models/' + MODEL_NAME + '.pickle'

random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

data_transform = transforms.Compose([
    transforms.Resize((50, 50)),
    transforms.ToTensor(),
    normalizer
])

train_dataset = datasets.ImageFolder(
    args.data_path,
    data_transform
)

train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)

encoder = Encoder()
mlp = MLP()


def evaluateInEnv():
    encoder.eval()
    mlp.eval()
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
        while not done:
            x = Image.fromarray(get_obs())
            inp = data_transform(x)
            output = encoder(inp[np.newaxis, :, :, :])
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
    encoder.train()
    mlp.train()
    return d

weights = [1., 1., 0.1, 0.01, 0.01, 0.01, 0.01, 0.01] # moving forward is quite likely
class_weights = torch.FloatTensor(weights)
cross_entropy = nn.CrossEntropyLoss(weight=class_weights)
params = list(encoder.parameters()) + list(mlp.parameters())
optimizer = optim.Adam(params, lr=args.lr)

model_metrics = ModelMetrics(MODEL_NAME)

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
    
    model_metrics.add_loss(avg_loss)

    if (epoch+1) % args.eval_epoch == 0:
        eval_result = evaluateInEnv()
        model_metrics.add_eval(epoch, eval_result) 
      
eval_result = evaluateInEnv()
model_metrics.add_eval(-1, eval_result) 

torch.save({
    'encoder_dict': encoder.state_dict(),
    'mlp_dict': mlp.state_dict()
}, MODEL_PATH)

with open(MODEL_METRICS_PATH, 'wb') as fp:
    pickle.dump(model_metrics.getDict(), fp)