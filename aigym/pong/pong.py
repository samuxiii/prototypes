import os
import gym
import random
import numpy as np
from Agent import Agent
from time import sleep


# code for the two only actions in Pong
UP_ACTION = 2
DOWN_ACTION = 3

# initializing our environment
env = gym.make("Pong-v0")

# beginning of an episode
observation = env.reset()

# model weights
h5file = "weights.h5"

# agent
agent = Agent()

# get model
if os.path.exists(h5file):
    agent.model.load_weights(h5file)

# training conf
training = True
# x_train, y_train, rewards = [], [], []
# reward_sum = 0

# main loop
for i in range(10000000):
    # predict action
    action = agent.act(observation)
    movement = UP_ACTION if action == 0 else DOWN_ACTION

    # do one step
    next_observation, reward, done, info = env.step(movement)

    # save the current observation
    agent.remember(observation, action, reward, next_observation, done)

    # update state
    observation = next_observation

    if reward != 0:
        if reward == 1:
            print("Win!!")
        else:
            print("Lose..")

        if training:
            win = True if reward == 1 else False
            agent.replay(win)
            agent.model.save_weights(h5file)

    if done:
        print("epsilon:{}".format(agent.epsilon))
        observation = env.reset()

