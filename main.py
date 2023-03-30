import os, sys
import gymnasium as gym
import time

import text_flappy_bird_gym

from Agents import Agent1,Agent2,GreedyAgent, QLearningAgent
from functions.utils import mc_prediction, td_prediction, plot_V

if __name__ == '__main__':

    # initiate environment
    env = gym.make('TextFlappyBird-v0', height = 15, width = 20, pipe_gap = 4)


    agent = QLearningAgent(epsilon = 0.1,alpha=0.1,gamma= 1.0,height = 15, width = 20)#GreedyAgent(V)
    agent.training(num_episodes=20000,env=env)


    obs = env.reset()

    print(obs)
    obs = obs[0]

    actions = []
    observations = []
    rewards = []

    # iterate
    while True:

        # Select next action
        action = agent.policy(obs)#env.action_space.sample() #agent.policy(obs)  # ## for an agent, action = agent.policy(observation)
        actions.append(action)
        print(action)

        # Appy action and return new observation of the environment
        obs, reward, done, _, info = env.step(action)
        observations.append(obs)
        rewards.append(reward)

        # Render the game
        os.system("clear")
        sys.stdout.write(env.render())
        time.sleep(0.2) # FPS

        # If player is dead break
        if done:
            break

    env.close()
print("actions",actions)
print("obs",observations)
print("rewards",rewards)