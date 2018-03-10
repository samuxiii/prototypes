import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from time import sleep



if __name__ == "__main__":

    env = gym.make('CartPole-v0')

    for i_episode in range(5):
        observation = env.reset()
        for t in range(100):
            env.render()
            print("observation (cart_pos, cart_vel, pole_ang, pol_vel):\n{}".format(observation))

            action = env.action_space.sample()
            print("action: {}".format(action))
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    
    sleep(2)
