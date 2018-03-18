import os
import time
import random
import gym
from frozenlake import FrozenLakeEnv
import numpy as np


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
            #update V[s]
            V[s] = v

        #print("Loss: {:.6f}".format(diff))
        if delta < theta:
            break

    return V

def obtain_q_from_v(env, V, s):
    gamma = 1.0
    q = np.zeros(env.nA)

    for a in range(env.nA):
        #using MDP getting info to calculate q
        for prob, next_state, reward, done in env.P[s][a]:
            q[a] += prob * (reward + gamma * V[next_state])

    return q

def policy_improvement(env, V):
    policy = np.zeros([env.nS, env.nA])

    for s in range(env.nS):
        q = obtain_q_from_v(env, V, s)
        #retrieve best actions in a list
        best_actions = np.argwhere(q == np.max(q)).flatten()
        #get a list of probabilities actions
        probs = np.sum([np.eye(env.nA)[i] for i in best_actions], axis=0)
        #normalize them to 1 and set
        policy[s] = probs/len(best_actions)

    return policy

def policy_iteration(env):
    gamma = 1.0
    theta = 1e-7
    policy = equiprobable_policy(env.nS, env.nA)

    while True:
        V = policy_evaluation(env, policy)
        new_policy = policy_improvement(env, V)

        #calculate difference between policies and cut when converge under theta
        nV = policy_evaluation(env, new_policy)
        delta = np.max(np.abs(V - nV))

        if delta < theta:
            break
        else:
            #update the policy
            policy = new_policy.copy()

        print("delta optimal policy: {}".format(delta))

    return policy, V

def getAction(policy, state):
    #get better action from policy
    print("Q(s,a): {}".format(policy[state]))
    print("action: {}".format(np.argmax(policy[state])))
    return np.argmax(policy[state])


def plot_values(V):
    # reshape value function
    import matplotlib.pyplot as plt
    V_sq = np.reshape(V, (8,8))
    
    # plot the state-value function
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(V_sq, cmap='cool')
    for (j,i),label in np.ndenumerate(V_sq):
        ax.text(i, j, np.round(label, 5), ha='center', va='center', fontsize=7)
    plt.tick_params(bottom='off', left='off', labelbottom='off', labelleft='off')
    plt.title('State-Value Function')
    plt.show()
    
def main():

    env = FrozenLakeEnv(map_name='8x8')

    policy, V = policy_iteration(env)
    print("V: {}".format(V))
    print("policy: {}".format(policy))
    #plot_values(V)
 
    time.sleep(1)

    #episodes
    for e in range(3):
        state = env.reset()
        while True:
            os.system('clear')
            env.render()
       
            print("\ncurrent state: {}".format(state))
            action = getAction(policy, state)
            next_state, reward, done, info = env.step(action)
        
            state = next_state
            time.sleep(0.2)
        
            #end
            if done:
                os.system('clear')
                env.render() #print last position
                print("\nFuck Yeah!!")
                time.sleep(2)
                break
        

if __name__ == "__main__":
    main()
