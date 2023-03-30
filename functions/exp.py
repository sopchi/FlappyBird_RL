import os, sys
import gymnasium as gym
import time

import text_flappy_bird_gym

from Agents import Agent1,Agent2,GreedyAgent, QLearningAgent,MCcontrolAgent
from functions.utils import mc_prediction, td_prediction, plot_V



# initiate environment
env = gym.make('TextFlappyBird-v0', height = 15, width = 20, pipe_gap = 4)
#V = td_prediction(env, Agent1(), ep=100, gamma=1.0, alpha=0.1)

#agent = QLearningAgent(env,epsilon = 0.1,alpha=0.1,gamma= 1.0,height = 15, width = 20)#GreedyAgent(V)
#agent.training(num_episodes=50000,env=env)
agent = MCcontrolAgent( env,height = 15, width = 20)
agent.mc_control(env,100000)

print("done training")

obs = env.reset()

obs = obs[0]

actions = []
observations = []
rewards = []
scores = [0]
# iterate
while scores[-1]<10000:

    # Select next action
    action = agent.policy(obs)#env.action_space.sample() #agent.policy(obs)  # ## for an agent, action = agent.policy(observation)
    actions.append(action)

    # Appy action and return new observation of the environment
    obs, reward, done, _, info = env.step(action)
    scores.append(info['score'])
    observations.append(obs)
    rewards.append(reward)

    # If player is dead break
    if done:
        break

env.close()

print("actions",actions)
print("obs",observations)
print("rewards",rewards)
print("scores",scores)