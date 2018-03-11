import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from time import sleep


class Agent:

    def __init__(self):
        self.memory = []
        pass

    def remember(self):
        #store in memory the different states, actions, rewards...
        pass

    def replay(self):
        #fit model from memory
        pass

    def act(self):
        #predict the action to do

        #temporary random action
        return random.randint(0,1)


if __name__ == "__main__":

    env = gym.make('CartPole-v0')
    agent = Agent()

    for i_episode in range(5):
        state = env.reset()

        for t in range(1,100):
            env.render()
            print("state (cart_pos, cart_vel, pole_ang, pol_vel):\n{}".format(state))

            #perform the action
            # 0->left, 1->right
            action = agent.act()
            print("action: {}".format(action))

            next_state, reward, done, info = env.step(action)

            #we save the current observation
            agent.remember(state, action, reward, next_state, done)

            #update state
            state = next_state

            #evaluate
            if done:
                print("Episode finished after {} timesteps".format(t))
                break

        #at the end of episode, train the model
        agent.replay()
   
    #before exit
    sleep(2)

