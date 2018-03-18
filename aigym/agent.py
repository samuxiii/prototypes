import os
import time
import random
import gym
from frozenlake import FrozenLakeEnv
import numpy as np

env = FrozenLakeEnv(map_name='8x8')

# print the state space and action space
print(env.nS)
print(env.nA)
time.sleep(1)

def equiprobable_policy(num_states, num_actions):
    return np.ones([num_states, num_actions]) / num_actions

#Iterative Policy Evaluation

def policy_evaluation(env, policy):
    gamma = 1.0
    theta = 1e-4 #max difference between old and new values
    states = range(env.nS)
    V = np.zeros(env.nS) #init

    while True:
        delta = 0

        for s in states:
            v = 0
            for a, action_prob in enumerate(policy[s]):
                #env.P has been exposed (MDP of frozenlake.py)
                for prob, next_state, reward, done in env.P[s][a]:
                    v += action_prob * prob * (reward + gamma * V[next_state])
            
            delta = max(delta, np.abs(v - V[s]))

        #print("Loss: {:.6f}".format(diff))
        if delta < theta:
            break

    return V


def getAction(state, V):
    #get better action
    return random.randint(0,3)

for i in range(100):
    os.system('clear')
    env.render()
    state = env.reset()

    policy = equiprobable_policy(env.nS, env.nA)
    V = policy_evaluation(env, policy)
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


