import torch
import torch.nn as nn
import torch.optim as optim
from agents.net import MLP, Encoder
import gym
import numpy as np
from PIL import Image

from torchvision import transforms as T

class ContextPredictionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.mlp_cp = MLP(dim=10368)
        self.mlp_bc = MLP(dim=50185)

    def forward(self, x):
        patch1, patch2 = x
        features1 = self.encoder(patch1)
        features2 = self.encoder(patch2)
        features = torch.cat((features1, features2))
        out = self.mlp(features)
        return out 

    def eval_in_env(self, env_name, transform, device, top_view=False, num_episodes=100, domain_rand=False):
        self.encoder.eval()
        self.mlp_bc.eval()
        
        env = gym.make(env_name)
        env.reset()
        get_obs = env.render_top_view if top_view else env.render_obs

        if domain_rand:
            env.domain_rand = True

        metric_steps = []
        metric_success = []

        for i in range(num_episodes):
            done = False
            steps = 0
            while not done and steps<=500:
                x = Image.fromarray(get_obs())
                prev_a = 8
                v = torch.zeros(9, dtype=torch.float).to(device); v[prev_a] = 1.0
                inp = transform(x).to(device)
                output = self.encoder(inp[np.newaxis, :, :, :])
                v = v[np.newaxis, :]
                mlp_input = torch.cat((output, v), axis=1)
                output = self.mlp_bc(mlp_input)
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
        self.encoder.train()
        self.mlp_bc.train()
        return d