import os
import gym
import random
import numpy as np
from Agent import Agent
from time import sleep

# code for the two only actions in Pong
UP_ACTION = 2
DOWN_ACTION = 3
NO_ACTION = 0

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

episode = 0
previousObs = np.zeros_like(observation)
wins = 0
# main loop
while episode < 10000:
    # predict action
    diffObs = observation - previousObs
    action = agent.act(diffObs)

    # movement = UP_ACTION if action == 1 else DOWN_ACTION
    movement = NO_ACTION
    if action == 1:
        movement = UP_ACTION
    elif action == 2:
        movement = DOWN_ACTION

        # do one step
    next_observation, reward, done, info = env.step(movement)

    # save the current observation
    agent.remember(diffObs, action, reward, next_observation, done)

    # update state
    previousObs = observation
    observation = next_observation

    if reward != 0:
        if reward == 1:
            wins += 1

        if training:
            agent.replay()
            agent.model.save_weights(h5file)

    if done:
        print("******* episode:{} wins:{} (epsilon:{}) ********".format(episode, wins, agent.epsilon))

        if wins >= 20:
            break

        # decrease exploration rate
        if agent.epsilon > 0.01:
            agent.epsilon *= 0.997

        observation = env.reset()
        episode += 1
        wins = 0
