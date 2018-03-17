import os
import time
import random
import gym
import numpy as np

env = gym.make('FrozenLake8x8-v0')

# print the state space and action space
print(env.observation_space.n)
print(env.action_space.n)
time.sleep(1)

def equiprobable_policy():
    states = 64
    actions = 4
    return np.ones([states, actions]) / actions

def certain_policy():
    states = 64
    actions = 4
    P = np.zeros([states, actions])

    for s in range(0, states):
       #corners
       if s == 0:
           P[s][0], P[s][1], P[s][2], P[s][3] = 0, 1/2, 1/2, 0
       elif s == 7:
           P[s][0], P[s][1], P[s][2], P[s][3] = 1/2, 1/2, 0, 0
       elif s == 56:
            P[s][0], P[s][1], P[s][2], P[s][3] = 0, 0, 1/2, 1/2
       elif s == 63:
            P[s][0], P[s][1], P[s][2], P[s][3] = 1/2, 0, 0, 1/2
       #sides
       elif s in [1,2,3,4,5,6]:
           P[s][0], P[s][1], P[s][2], P[s][3] = 1/3, 1/3, 1/3, 0
       elif s in [57,58,59,60,61,62]:
           P[s][0], P[s][1], P[s][2], P[s][3] = 1/3, 0, 1/3, 1/3
       elif s in [8,16,24,32,40,48]:
           P[s][0], P[s][1], P[s][2], P[s][3] = 0, 1/3, 1/3, 1/3
       elif s in [15,23,31,39,47,55]:
           P[s][0], P[s][1], P[s][2], P[s][3] = 1/3, 1/3, 0, 1/3
       #other
       else:
           P[s][0], P[s][1], P[s][2], P[s][3] = 1/4, 1/4, 1/4, 1/4

    return P

#Iterative Policy Evaluation
def getClosest(state):
    #return left, down, right, up
    #corners
    if state == 0:
        return 0, 8,  1, 0
    elif s == 7:
        return 6,  15, 0, 0
    elif s == 56:
        return 0, 0, 57, 48
    elif s == 63:
        return 62, 0, 0,  55
    #sides
    elif s in [1,2,3,4,5,6]:
        return state-1, state+8, state+1, 0
    elif s in [57,58,59,60,61,62]:
        return state-1, 0, state+1, state-8
    elif s in [8,16,24,32,40,48]:
        return 0, state+8, state+1, state-8
    elif s in [15,23,31,39,47,55]:
        return state-1, state+8, 0, state-8
    #other
    else:
        return state-1, state+8, state+1, state-8


def policy_evaluation(num_states, policy):
    gamma = 1.0
    theta = 1e-4 #max difference between old and new values
    states = range(0, num_states)
    V = np.random.rand(64) #init

    while True:
        diff = 0

        for s in states:
            v = V[s]
            left, down, right, up = getClosest(state)

            V[s] = (policy[s][0] * V[left] +
                    policy[s][1] * V[down] +
                    policy[s][2] * V[right] +
                    policy[s][3] * V[up])

            #
            diff = max(diff, np.abs(v - V[s]))

        #print("Loss: {:.6f}".format(diff))
        if diff < theta:
            break

    return V


def getAction(state, V):
    #get better action
    left, down, right, up = getClosest(state)
    return np.argmax([V[left], V[down], V[right], V[up]]) 

for i in range(10):
    os.system('clear')
    env.render()
    state = env.reset()

    policy = equiprobable_policy()
    #policy = certain_policy()
    V = policy_evaluation(env.observation_space.n, policy)
    print("V: {}".format(V))
    action = getAction(state, V)

    next_state, reward, done, info = env.step(action)

    state = next_state
    time.sleep(0.5)

    #end
    if done:
        os.system('clear')
        env.render() #print last position
        time.sleep(2)
        break


