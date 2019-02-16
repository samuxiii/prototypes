import random
import numpy as np
from keras import backend as K
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam


class Agent:

    def __init__(self):
        self.memory = []
        self.epsilon = 1.0  # exploration rate
        self.model = self.__model()

    def __model(self):
        model = Sequential()
        model.add(Conv2D(16, kernel_size=8, strides=4, activation='relu', input_shape=(80, 80, 1)))
        model.add(BatchNormalization())
        model.add(Conv2D(32, kernel_size=4, strides=2, activation='relu'))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        return model

    def preprocess(self, I):
        # prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector
        I = I[35:195]  # crop
        I = I[::2, ::2, 0]  # downsample by factor of 2
        I[I == 144] = 0  # erase background (background type 1)
        I[I == 109] = 0  # erase background (background type 2)
        I[I != 0] = 1  # everything else (paddles, ball) just set to 1
        I = np.reshape(I, (80, 80, 1))
        return I  # shape:(80, 80, 1)

    def discount_rewards(self, r, gamma=0.99):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, len(r))):
            if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        # normalize
        discounted_r -= np.mean(discounted_r)
        discounted_r /= np.std(discounted_r)
        return discounted_r

    def remember(self, state, action, reward, next_state, done):
        # states must be preprocessed
        if (state.shape[0] != 80 and state.shape[1] != 80):
            state = self.preprocess(state)
        if (next_state.shape[0] != 80 and next_state.shape[1] != 80):
            next_state = self.preprocess(next_state)

        # store in memory the different states, actions, rewards...
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        # fit model from memory
        gamma = 0.5  # importance of the next reward

        # initialize
        list_x_batch, list_y_batch = [], []

        # get the list of rewards
        _, _, list_r_batch, _, _ = zip(*self.memory)

        # print("steps:{}".format(len(self.memory)))
        for state, action, reward, next_state, done in self.memory:
            state = np.expand_dims(state, axis=0)

            target = np.zeros([3])  # 0's for up and down => [0, 0]
            target[action] = 1  # performed action is set to 1

            # append
            list_x_batch.append(state)
            list_y_batch.append(target)

        # clean
        self.memory = []

        # train the model
        x_batch = np.vstack(list_x_batch)
        y_batch = np.vstack(list_y_batch)
        r_batch = self.discount_rewards(list_r_batch)

        self.model.fit(x_batch, y_batch, sample_weight=r_batch, verbose=0)

    def act(self, state):
        # preprocess the sample
        state = self.preprocess(state)

        if self.epsilon > np.random.rand():
            return random.randint(0, 2)

        # predict the action to do
        state = np.expand_dims(state, axis=0)
        action_values = self.model.predict(state)
        # print("Predictions:{}".format(action_values))

        return np.argmax(action_values)