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

# mapping actions: model output -> environment
action2move = {0:NO_ACTION, 1:UP_ACTION, 2:DOWN_ACTION}

# initializing our environment
env = gym.make("Pong-v0")

# beginning of an episode
observation = env.reset()
previousObs = np.zeros([80,80,1])

# model weights
h5file = "weights.h5"

# agent
agent = Agent()

# get model
if os.path.exists(h5file):
    agent.model.load_weights(h5file)
else:
    print("Not weights found. Exit.")
    exit(0)

#
episode = 1
wins = 0
wins_list = []
win_running_mean = 0

# main loop
for i in range(100000):
    env.render()
    sleep(0.005)

    # predict action
    observation = agent.preprocess(observation)
    diffObs = observation - previousObs
    
    action, actions = agent.act(diffObs)
    move = action2move[action]     

    # do one step
    next_observation, reward, done, _ = env.step(move)

    # update state
    previousObs = observation
    observation = next_observation

    if reward != 0:
        if reward == 1:
            wins += 1

    if done:
        win_running_mean += (wins - win_running_mean)/(len(wins_list)+1)
        wins_list.append(win_running_mean)
        print("******* episode:{} wins:{} perf:{:.3f} ********".format(episode, wins, win_running_mean))

        observation = env.reset()
        previousObs = np.zeros([80,80,1])
        wins = 0
        episode += 1

