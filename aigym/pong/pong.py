import os
import gym
import pickle
import random
import numpy as np
from Agent import Agent
from time import sleep

#!rm weights.h5
#!rm data.pkl

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
previousObs = np.zeros_like(observation)

# model weights
h5file = "weights.h5"

# agent
agent = Agent()

# try to load previous model
if os.path.exists(h5file):
    agent.model.load_weights(h5file)

# training conf
training = False

# main loop
episode = 0
wins = 0
win_running_mean = 0
wins_list = []

while episode < 10000: 
    # predict action
    diffObs = observation - previousObs
    
    action, actions = agent.act(diffObs)
    move = action2move[action]     

    # do one step
    next_observation, reward, done, _ = env.step(move)

    # save the current observation
    agent.remember(diffObs, action, actions, reward)

    # update state
    previousObs = observation
    observation = next_observation

    
    if reward != 0:
        if reward == 1:
            wins += 1
            
        if training and (episode % 10 == 0):
            agent.replay()
            
    if done:
        
        episode += 1
        win_running_mean += (wins - win_running_mean)/(len(wins_list)+1)
        wins_list.append(win_running_mean)
        print("******* episode:{} wins:{} perf:{:.3f} ********".format(episode, wins, win_running_mean))
        
        if episode % 50 == 0:
          agent.model.save_weights(h5file)
          with open('data.pkl', 'wb') as f:
            pickle.dump(wins_list, f)
          
        observation = env.reset()
        wins = 0


