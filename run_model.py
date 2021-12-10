"""
This script allows you to run a trained model interactively on an env
"""
import argparse
import pyglet
import math
import numpy as np
import gym
import gym_miniworld
import torch
from torchvision import transforms
from PIL import Image

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

prev_a = 8

# normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                # std=[0.229, 0.224, 0.225])

# encoder = Encoder()
# mlp = MLP()
# trsf = transforms.Compose([
#     transforms.Resize((50, 50)),
#     transforms.ToTensor(),
#     normalizer
# ])
transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor()]
)

# load model
model = torch.load(args.model_path)
# encoder.load_state_dict(model_checkpoint['encoder_dict'])
# mlp.load_state_dict(model_checkpoint['mlp_dict'])

# encoder.eval()
# mlp.eval()

# Create the display window
env.render('pyglet', view=view_mode)

def agent_step(t):
    global prev_a
    prev_output = torch.zeros(1, 9)
    prev_output[0, prev_a] = 1
    x = Image.fromarray(get_obs())
    inp = transform(x)
    features = model.encoder(inp.unsqueeze(0))
    features = torch.cat((features, prev_output), dim=1)
    output = model.mlp_bc(features)
    output = output.clone().detach()
    action = output[0].argmax().item()

    obs, reward, done, info = env.step(action)

    if reward > 0:
        print('reward={:.2f}'.format(reward))
    
    if done:
        print('done!')
        env.reset()
    prev_a = action
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