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
        state = env.reset()

        for t in range(1,100):
            env.render()
            print("state (cart_pos, cart_vel, pole_ang, pol_vel):\n{}".format(state))

            # 0->left, 1->right
            action = env.action_space.sample()
            print("action: {}".format(action))

            next_state, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t))
                break
    
    sleep(2)
