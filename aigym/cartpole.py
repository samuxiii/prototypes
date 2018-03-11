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
        self.model = self.__model()
        pass

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
        size = 16 if len(self.memory) > 16 else len(self.memory)
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


    def act(self, state):
        #predict the action to do
        action_values = self.model.predict(state)

        return np.argmax(action_values[0])


if __name__ == "__main__":

    env = gym.make('CartPole-v0')
    agent = Agent()

    for i_episode in range(5):
        state = env.reset()

        for t in range(1,100):
            env.render()
            print("state (cart_pos, cart_vel, pole_ang, pol_vel):\n{}".format(state))

            #row vector
            state = state.reshape(1, -1)

            #perform the action
            # 0->left, 1->right
            action = agent.act(state)
            print("action: {}".format(action))

            next_state, reward, done, info = env.step(action)

            #row vector
            next_state = next_state.reshape(1, -1)

            #save the current observation
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

