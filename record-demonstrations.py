"""
This script is used to record demonstrations (using the keyboard arrows) from user
and storing them in specified format
"""

import sys
import argparse
import pyglet
import math
from pyglet.window import key
from pyglet import clock
import numpy as np
import gym
import gym_miniworld

from core.demo import Demonstration
from core.utils import store_demo

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default='MiniWorld-Hallway-v0')
parser.add_argument('--domain_rand', action='store_true', help='enable domain randomization')
parser.add_argument('--top_view', action='store_true', help='show the top view instead of the agent view')
parser.add_argument('--store_depth', action='store_true', help='store depth maps in the demonstrations')
args = parser.parse_args()

env = gym.make(args.env_name)

if args.domain_rand:
    env.domain_rand = True

view_mode = 'top' if args.top_view else 'agent'

demo = Demonstration()

# get_obs function - (60, 80) image
get_obs = env.render_top_view if args.top_view else env.render_obs
get_dm = env.render_depth

env.reset()

obs, dm = None, None

def update_obs():
    global obs, dm
    obs = get_obs()
    if args.store_depth:
        dm = get_dm()
update_obs()

# Create the display window
env.render('pyglet', view=view_mode)

def step(action):
    global obs, dm, env
    print('step {}/{}: {}'.format(env.step_count+1, env.max_episode_steps, env.actions(action).name))
    demo.add(obs, action, dm)
    obs, reward, done, info = env.step(action)

    if reward > 0:
        print('reward={:.2f}'.format(reward))

    if done:
        print('done!')
        store_demo(demo, args.env_name, view_mode)
        env.reset()    

    update_obs()
    env.render('pyglet', view=view_mode)

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        demo.clear()
        env.reset() # doesn't store demonstration here
        update_obs()
        env.render('pyglet', view=view_mode)
        return

    if symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

    if symbol == key.UP:
        step(env.actions.move_forward)
    elif symbol == key.DOWN:
        step(env.actions.move_back)

    elif symbol == key.LEFT:
        step(env.actions.turn_left)
    elif symbol == key.RIGHT:
        step(env.actions.turn_right)

    elif symbol == key.PAGEUP or symbol == key.P:
        step(env.actions.pickup)
    elif symbol == key.PAGEDOWN or symbol == key.D:
        step(env.actions.drop)

    elif symbol == key.ENTER:
        step(env.actions.done)

@env.unwrapped.window.event
def on_key_release(symbol, modifiers):
    pass

@env.unwrapped.window.event
def on_draw():
    env.render('pyglet', view=view_mode)

@env.unwrapped.window.event
def on_close():
    pyglet.app.exit()

# Enter main event loop
pyglet.app.run()

env.close()