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
        self.epsilon = 1.0 #exploration rate
        self.model = self.__model()

    def __model(self):
        features = 4
        learning_rate = 0.01

        model = Sequential()
        model.add(Dense(2, input_dim=features, activation='relu'))

        #the loss function will be MSE between the action-value Q
        model.compile(loss='mse', optimizer=Adam(lr=learning_rate))

        return model

    def remember(self, state, action, reward, next_state, done):
        #store in memory the different states, actions, rewards...
        self.memory.append( (state, action, reward, next_state, done) )

    def replay(self):
        #fit model from memory
        gamma = 0.90

        #take care the memory could be big, so using minibatch
        size = 32 if len(self.memory) > 32 else len(self.memory)
        minibatch = random.sample(self.memory, size)

        for state, action, reward, next_state, done in minibatch:

            if not done: #calculate discounted reward
                action_values = self.model.predict(next_state)[0]
                #following the formula of action-value expectation
                reward = reward + gamma * np.amax(action_values)

            target = self.model.predict(state)

            #customize the reward for this certain action using 
            #the one found in the memory
            target[0][action] = reward

            #train the model
            self.model.fit(state, target, epochs=1, verbose=0)

        #decrease exploration rate
        if self.epsilon > 0.01:
            self.epsilon *= 0.99


    def act(self, state):
        if self.epsilon > np.random.rand():
            return random.randint(0,1)

        #predict the action to do
        action_values = self.model.predict(state)

        return np.argmax(action_values[0])


if __name__ == "__main__":

    env = gym.make('CartPole-v0')
    agent = Agent()
    
    solved = False

    for episode in range(1000):
        state = env.reset()
        total_reward = 0

        for step in range(1,200):
            #env.render()
            #print("state (cart_pos, cart_vel, pole_ang, pol_vel):\n{}".format(state))

            #row vector
            state = state.reshape(1, -1)

            #perform the action
            # 0->left, 1->right
            action = agent.act(state)
            #print("action: {}".format(action))

            next_state, reward, done, info = env.step(action)

            #row vector
            next_state = next_state.reshape(1, -1)

            #save the current observation
            agent.remember(state, action, reward, next_state, done)

            #update state
            state = next_state

            #update total reward
            total_reward += reward

            #evaluate
            if done:
                print("Episode: {} Reward: {} Epsilon: {:.3f}".format(episode, total_reward, agent.epsilon))
                break

        #solved when reward >= 195 before 100 episodes
        if total_reward > 195:
            print("SOLVED!!!")
            break

        #at the end of episode, train the model
        agent.replay()
   
    #before exit
    sleep(2)

