import argparse
import math
import numpy as np
import gym
import gym_miniworld
import torch
from torchvision import transforms
from PIL import Image
import pyglet
from agents.net import MLP, Encoder
import os
from pathlib import Path
import pickle
from time import time
from gym import wrappers


parser = argparse.ArgumentParser()
parser.add_argument('--env_name', default='MiniWorld-OneRoom-v0')
parser.add_argument('--domain_rand', action='store_true', help='enable domain randomization')
parser.add_argument('--no_time_limit', action='store_true', help='ignore time step limits')
parser.add_argument('--top_view', action='store_true', help='show the top view instead of the agent view')
parser.add_argument('--model_path', default='models/bc.pt')
parser.add_argument('--time_limit', default=250,  type=int, help='Maximum episode steps')
parser.add_argument('--gpu', action='store_true', help='Uses a gpu when possible')
parser.add_argument("--record_video", action='store_true', help="records video")
parser.add_argument("--vid_save_path", default=".", help="where to save the video")
parser.add_argument("--max_episodes", type=float, default=float('inf'), help="exits after running this amount of episodes.")
parser.add_argument("--fps", type=float, default=60, help="Max FPS.")
parser.add_argument("--no_graphics", action='store_true', help='run without graphics')
args = parser.parse_args()

device='cpu'
if args.gpu:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

prev_a = 8

steps = 0


env = gym.make(args.env_name)
# env = gym.wrappers.Monitor(env, "./vid", video_callable=lambda episode_id: True,force=True)
if args.record_video:
    env = wrappers.Monitor(env, os.path.join(args.vid_save_path, str(time()) + '/'),
                           video_callable=lambda episode_id: True,force=True)
env.reset()
get_obs = env.render_top_view if args.top_view else env.render_obs

env.max_episode_steps

if args.no_time_limit:
    env.max_episode_steps = math.inf
else:
    env.max_episode_steps = args.time_limit
if args.domain_rand:
    env.domain_rand = True

view_mode = 'top' if args.top_view else 'agent'

normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

encoder = Encoder().to(device)
mlp = MLP(dim=2304+9).to(device)
trsf = transforms.Compose([
    transforms.RandomResizedCrop(50, scale=(0.2, 1.)),
    transforms.ToTensor(),
    normalizer
])

# load model
model_checkpoint = torch.load('test_simsiam.pt')
encoder.load_state_dict(model_checkpoint['encoder_dict'])
mlp.load_state_dict(model_checkpoint['mlp_dict'])

encoder.eval()
mlp.eval()

metric_steps = []
metric_success = []
steps = 0
prev_a = 8

if args.no_graphics:
    for i in range(int(args.max_episodes)):
        done = False
        steps = 0
        while not done:
            x = Image.fromarray(get_obs())
            inp = trsf(x)
            output = encoder(inp[np.newaxis, :, :, :].to(device))
            prev_a_v = torch.zeros((1, 9)).to(output.device)
            prev_a_v[0, prev_a] = 1.0
            output = torch.cat((output, prev_a_v), dim=-1)
            output = mlp(output)
            output = output.clone().detach()
            action = output[0].argmax().item()

            _, reward, done, _ = env.step(action)
            steps += 1
            prev_a = action

            if reward > 0:
                print('#{} reward={:.2f}'.format(i, reward))

            
            if done:
                env.reset()
                metric_success.append(reward > 0)
                metric_steps.append(steps)
                steps = 0
                prev_a = 8

else:
    # Create the display window
    env.render('pyglet', view=view_mode)

    # prev_a = 8

    # steps = 0

    def agent_step(t):
        x = Image.fromarray(get_obs())
        inp = trsf(x)
        output = encoder(inp[np.newaxis, :, :, :].to(device))
        prev_a_v = torch.zeros((1, 9)).to(output.device)
        prev_a_v[0, agent_step.prev_a] = 1.0
        output = torch.cat((output, prev_a_v), dim=-1)
        output = mlp(output)
        output = output.clone().detach()
        action = output[0].argmax().item()

        _, reward, done, _ = env.step(action)
        agent_step.steps += 1
        agent_step.prev_a = action

        if reward > 0:
            print('#{} reward={:.2f}'.format(agent_step.episodes, reward))

        
        if done:
            env.reset()
            metric_success.append(reward > 0)
            metric_steps.append(agent_step.steps)
            agent_step.steps = 0
            agent_step.prev_a = 8
            agent_step.episodes += 1
            if agent_step.episodes >= args.max_episodes:
                print('Stopping...')
                pyglet.app.exit()
        env.render('pyglet', view=view_mode)

    @env.unwrapped.window.event
    def on_key_release(symbol, modifiers):
        pass

    @env.unwrapped.window.event
    def on_draw():
        # global prev_a
        # global steps
        # steps = 0
        # prev_a = 8
        env.render('pyglet', view=view_mode)

    @env.unwrapped.window.event
    def on_close():
        pyglet.app.exit()

    agent_step.steps = 0
    agent_step.prev_a = 8
    agent_step.episodes = 0
    pyglet.clock.schedule_interval(agent_step, 1./args.fps)


    # Enter main event loop
    pyglet.app.run()

    env.close()

d = {
    'success_rate': np.mean(metric_success),
    'metric_steps': metric_steps,
    'mean_steps': np.mean(metric_steps)
}

print(d)
m = Path(args.model_path).parts[-1].split('.')[0] + '.pickle'
with open('results/' + m, 'wb') as fp:
    pickle.dump(d, fp)
env.close()

print(d)

















# NUM_EPISODES = 100

# metric_steps = []
# metric_success = []


# with torch.no_grad():
#     for i in range(NUM_EPISODES):
#         prev_a = 8
#         done = False
#         steps = 0
#         while not done:
#             x = Image.fromarray(get_obs())
#             inp = trsf(x)
#             output = encoder(inp[np.newaxis, :, :, :].to(device))
#             prev_a_v = torch.zeros((1, 9)).to(output.device)
#             prev_a_v[0, prev_a] = 1.0
#             output = torch.cat((output, prev_a_v), dim=-1)
#             output = mlp(output)
#             output = output.clone().detach()
#             action = output[0].argmax().item()

#             _, reward, done, _ = env.step(action)
#             steps += 1
#             prev_a = action

#             if reward > 0:
#                 print('reward={:.2f}'.format(reward))
    
            
#             if done:
#                 env.reset()
#                 metric_success.append(reward > 0)
#                 metric_steps.append(steps)
#             env.render('pyglet', view=view_mode)

# # m = Path(args.model_path).parts[-1].split('.')[0] + '.pickle'
# # with open('results/' + m, 'wb') as fp:
# #     d = {
# #         'success_rate': np.mean(metric_success),
# #         'metric_steps': metric_steps
# #     }
# #     pickle.dump(d, fp)
# # env.close()
# d = {
#         'success_rate': np.mean(metric_success),
#         'metric_steps': metric_steps
#     }
# print(d)