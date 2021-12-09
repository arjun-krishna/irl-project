"""
This script allows you to run a trained behavior clone model interactively on an env
"""
import argparse
import math
import numpy as np
import gym
import gym_miniworld
import torch
from torchvision import transforms
from PIL import Image

from agents.net import MLP, Encoder
import os
from pathlib import Path
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default='MiniWorld-Hallway-v0')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--no-time-limit', action='store_true', help='ignore time step limits')
parser.add_argument('--top_view', action='store_true', help='show the top view instead of the agent view')
parser.add_argument('--model_path', default='models/bc.pt')
args = parser.parse_args()

env = gym.make(args.env_name)
env.reset()
get_obs = env.render_top_view if args.top_view else env.render_obs

if args.no_time_limit:
    env.max_episode_steps = math.inf
if args.domain_rand:
    env.domain_rand = True

view_mode = 'top' if args.top_view else 'agent'

normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

encoder = Encoder()
mlp = MLP()
trsf = transforms.Compose([
    transforms.RandomResizedCrop(50, scale=(0.2, 1.)),
    transforms.ToTensor(),
    normalizer
])

# load model
model_checkpoint = torch.load(args.model_path)
encoder.load_state_dict(model_checkpoint['encoder_dict'])
mlp.load_state_dict(model_checkpoint['mlp_dict'])

encoder.eval()
mlp.eval()

NUM_EPISODES = 100

metric_steps = []
metric_success = []

for i in range(NUM_EPISODES):
    done = False
    steps = 0
    while not done:
        x = Image.fromarray(get_obs())
        inp = trsf(x)
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

m = Path(args.model_path).parts[-1].split('.')[0] + '.pickle'
with open('results/' + m, 'wb+') as fp:
    d = {
        'success_rate': np.mean(metric_success),
        'metric_steps': metric_steps
    }
    pickle.dump(d, fp)
env.close()
print(d)