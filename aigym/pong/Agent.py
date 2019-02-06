import random
import numpy as np
from keras import backend as K
from keras.layers import Dense, Conv2D, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

class Agent:

    def __init__(self):
        self.memory = []
        self.epsilon = 1.0  # exploration rate
        self.model = self.__model()

    def __model(self):
        model = Sequential()
        model.add(Conv2D(16, kernel_size=8, strides=4, activation='relu', input_shape=(80, 80, 3)))
        model.add(Conv2D(32, kernel_size=4, strides=2, activation='relu'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def preprocess(self, I):
        # prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector
        I = I[35:195]  # crop
        I = I[::2, ::2, :]  # downsample by factor of 2
        I[I == 144] = 0  # erase background (background type 1)
        I[I == 109] = 0  # erase background (background type 2)
        I[I != 0] = 1  # everything else (paddles, ball) just set to 1
        return I  # shape:(80, 80, 3)

    def remember(self, state, action, reward, next_state, done):
        # states must be preprocessed
        if (state.shape[0] != 80 and state.shape[1] != 80):
            state = self.preprocess(state)
        if (next_state.shape[0] != 80 and next_state.shape[1] != 80):
            next_state = self.preprocess(next_state)

        # store in memory the different states, actions, rewards...
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, win):
        # fit model from memory
        gamma = 0.5  # importance of the next reward
        max_batch_size = 512

        # take care the memory could be big, so using minibatch
        # minibatch = random.sample(self.memory, min(max_batch_size, len(self.memory)))
        minibatch = self.memory
        num_steps = len(minibatch)

        list_x_batch, list_y_batch = [], []
        step = 0
        print("steps:{}".format(num_steps))
        for state, action, reward, next_state, done in minibatch:

            state = np.expand_dims(state, axis=0)
            target = self.model.predict(state)[0]

            # if not done: #calculate discounted reward
            #    action_values = self.model.predict(next_state)[0]
            # following the formula of action-value expectation
            #    reward = reward + gamma * np.amax(action_values)

            # customize the obtained reward with the calculated
            #
            # reward for win  : 1/5  2/5  3/5  4/5  5/5
            #
            step += 1
            reg_win = (step / num_steps)
            reg_loose = 0 #-(step / num_steps)

            target[action] = reg_win if win else reg_loose

            # append
            list_x_batch.append(state)
            list_y_batch.append(target)

            # clean
            self.memory = []

        # train the model
        x_batch = np.vstack(list_x_batch)
        y_batch = np.vstack(list_y_batch)
        self.model.fit(x_batch, y_batch, verbose=0)

        # decrease exploration rate
        if self.epsilon > 0.01:
            self.epsilon *= 0.9997

    def act(self, state):
        # preprocess the sample
        state = self.preprocess(state)

        if self.epsilon > np.random.rand():
            return random.randint(0, 1)

        # predict the action to do
        state = np.expand_dims(state, axis=0)
        action_values = self.model.predict(state)

        return np.argmax(action_values)