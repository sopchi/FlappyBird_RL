from Agents import Agent1,Agent2,GreedyAgent
from utils import mc_prediction, td_prediction, plot_V

import os, sys
import gymnasium as gym
import time

import text_flappy_bird_gym

import random as rd 
import matplotlib as plt
import numpy as np
import sys

from collections import defaultdict

env = gym.make('TextFlappyBird-v0', height = 15, width = 20, pipe_gap = 4)
agent = Agent1()
V = td_prediction(env, agent, ep=1000, gamma=1.0, alpha=0.1)
#V= mc_prediction(env,agent, num_episodes = 2000, discount_factor=1.0)

agent = GreedyAgent(V)

obs = env.reset()[0]
while True:
    
    # Select next action
    print([state for state in V.keys() if state[0]==obs[0]+1])
    action = agent.policy(obs)
    print(action) #env.action_space.sample()#agent.policy(obs)  # ## for an agent, action = agent.policy(observation)

    # Appy action and return new observation of the environment
    obs, reward, done, _, info = env.step(action)
    if done:
        break

#plot_V(V)


print([state[1] for state in V.keys()])