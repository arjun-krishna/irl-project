"""
This script allows you to manually control the simulator
using the keyboard arrows.
"""

import argparse
import pyglet
import math
import numpy as np
import gym
import gym_miniworld
import torch
from torchvision import transforms

from agents.net import MLP, Encoder
from core.transforms import NormalizeTensor, RandomCrop, ToTensor

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

encoder = Encoder()
mlp = MLP()
trsf = transforms.Compose([
    RandomCrop(50),
    ToTensor(),
    NormalizeTensor()
])

# load model
model_checkpoint = torch.load(args.model_path)
encoder.load_state_dict(model_checkpoint['encoder_dict'])
mlp.load_state_dict(model_checkpoint['mlp_dict'])
encoder.eval()
mlp.eval()

# Create the display window
env.render('pyglet', view=view_mode)

def agent_step(t):
    sample = {'obs': get_obs(), 'action': -1}
    inp = trsf(sample)['obs']
    output = encoder(inp[np.newaxis, :, :, :])
    output = mlp(output)
    output = output.clone().detach()
    action = output[0].argmax().item()

    obs, reward, done, info = env.step(action)

    if reward > 0:
        print('reward={:.2f}'.format(reward))
    
    if done:
        print('done!')
        env.reset()

    env.render('pyglet', view=view_mode)

@env.unwrapped.window.event
def on_key_release(symbol, modifiers):
    pass

@env.unwrapped.window.event
def on_draw():
    env.render('pyglet', view=view_mode)

@env.unwrapped.window.event
def on_close():
    pyglet.app.exit()

pyglet.clock.schedule_interval(agent_step, 1./60)


# Enter main event loop
pyglet.app.run()

env.close()