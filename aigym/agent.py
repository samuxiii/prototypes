import os
import time
import random
import gym
from frozenlake import FrozenLakeEnv
import numpy as np

env = FrozenLakeEnv(map_name='8x8')

# print the state space and action space
print(env.observation_space.n)
print(env.action_space.n)
time.sleep(1)

def equiprobable_policy():
    states = 64
    actions = 4
    return np.ones([states, actions]) / actions

#Iterative Policy Evaluation

def policy_evaluation(num_states, policy):
    gamma = 1.0
    theta = 1e-4 #max difference between old and new values
    states = range(0, num_states)
    V = np.random.rand(64) #init

    while True:
        diff = 0

        for s in states:
            v = V[s]
            #TODO
            #
            diff = max(diff, np.abs(v - V[s]))

        #print("Loss: {:.6f}".format(diff))
        if diff < theta:
            break

    return V


def getAction(state, V):
    #get better action
    return random.randint(0,3)

for i in range(100):
    os.system('clear')
    env.render()
    state = env.reset()

    policy = equiprobable_policy()
    V = policy_evaluation(env.observation_space.n, policy)
    print("V: {}".format(V))
    action = getAction(state, V)

    next_state, reward, done, info = env.step(action)

    state = next_state
    time.sleep(0.2)

    #end
    if done:
        os.system('clear')
        env.render() #print last position
        time.sleep(2)
        break


