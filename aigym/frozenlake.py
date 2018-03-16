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

def policy():
    states = 64
    actions = 4
    P = np.array([states, actions])

    for s in range(0, states):
       #corners
       if s == 0:
           P[s]0], P[s][1], P[s][2], P[s][3] = 0, 1/2, 1/2, 0
       elif s == 7:
           P[s]0], P[s][1], P[s][2], P[s][3] = 1/2, 1/2, 0, 0
       elif s == 56:
            P[s]0], P[s][1], P[s][2], P[s][3] = 0, 0, 1/2, 1/2
       elif s == 63:
            P[s]0], P[s][1], P[s][2], P[s][3] = 1/2, 0, 0, 1/2
       #sides
       elif s in [1,2,3,4,5,6]:
           P[s]0], P[s][1], P[s][2], P[s][3] = 1/3, 1/3, 1/3, 0
       elif s in [57,58,59,60,61,62]:
           P[s]0], P[s][1], P[s][2], P[s][3] = 1/3, 0, 1/3, 1/3
       elif s in [8,16,24,32,40,48]:
           P[s]0], P[s][1], P[s][2], P[s][3] = 0, 1/3, 1/3, 1/3
       elif s in [15,23,31,39,47,55]:
           P[s]0], P[s][1], P[s][2], P[s][3] = 1/3, 1/3, 0, 1/3
       #other
       else:
           P[s]0], P[s][1], P[s][2], P[s][3] = 1/4, 1/4, 1/4, 1/4

    return P

#Iterative Policy Evaluation
def policy_evaluation(num_states):
    gamma = 1.0
    theta = 1e-4 #max difference between old and new values
    states = range(0, num_states)
    V = np.random.rand(64) #init

    while True:
        diff = 0

        for s in states:
            v = V[s]

            #print("state: {}".format(s))
            #corners
            if s == 0:
                #pl, pd, pr, pu = 0, 1/2, 1/2, 0
                V[s] = 1/2 * (V[8] + V[1])
            elif s == 7:
                #pl, pd, pr, pu = 1/2, 1/2, 0, 0
                V[s]  = 1/2 * (V[6] + V[15])
            elif s == 56:
                #pl, pd, pr, pu = 0, 0, 1/2, 1/2
                V[s] = 1/2 * (V[48] + V[57])
            elif s == 63:
                #pl, pd, pr, pu = 1/2, 0, 0, 1/2
                V[s] = 1/2 * (V[62] + V[55])
            #sides
            elif s in [1,2,3,4,5,6]:
                #pl, pd, pr, pu = 1/3, 1/3, 1/3, 0
                V[s] = 1/3 * (V[s+8] + V[s+1] + V[s-1])
            elif s in [57,58,59,60,61,62]:
                #pl, pd, pr, pu = 1/3, 0, 1/3, 1/3
                V[s] = 1/3 * (V[s-8] + V[s+1] + V[s-1])
            elif s in [8,16,24,32,40,48]:
                #pl, pd, pr, pu = 0, 1/3, 1/3, 1/3
                V[s] = 1/3 * (V[s-8] + V[s+1] + V[s+8])
            elif s in [15,23,31,39,47,55]:
                #pl, pd, pr, pu = 1/3, 1/3, 0, 1/3
                V[s] = 1/3 * (V[s-8] + V[s-1] + V[s+8])
            #other
            else:
                #pl, pd, pr, pu = 1/4, 1/4, 1/4, 1/4
                V[s] = 1/4 * (V[s-8] + V[s+8] + V[s-1] + V[s+1])

            #
            diff = max(diff, np.abs(v - V[s]))

        #print("Loss: {:.6f}".format(diff))
        if diff < theta:
            break

    return V


def getAction(state, V):
    #TODO 
    return random.randint(0,3)

for i in range(10):
    os.system('clear')
    env.render()
    state = env.reset()

    V = policy_evaluation(env.observation_space.n)
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
