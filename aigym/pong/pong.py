import gym
import random
import numpy as np
from time import sleep

def prepro(I):
# prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

# code for the two only actions in Pong
UP_ACTION = 2
DOWN_ACTION = 3

# initializing our environment
env = gym.make("Pong-v0")

# beginning of an episode
observation = env.reset()

# main loop
for i in range(300):
    env.render()
    action = random.randint(UP_ACTION, DOWN_ACTION)
    observation, reward, done, info = env.step(action)

    if done:
        env.reset()

    ##
    sleep(0.03)
    print(sum(prepro(observation)))
