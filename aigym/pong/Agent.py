import random
import numpy as np
from keras import backend as K
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, Activation, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam

class Agent:

    def __init__(self):
        self.memory = []
        self.model = self.__model()

    def __model(self, lr=0.001):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=3, strides=3, input_shape=(80, 80, 1), use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Flatten())
        model.add(Dense(256, use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Dense(3, activation='softmax'))
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=lr), metrics=['accuracy'])

        model.summary()

        return model

    def preprocess(self, I):
        # prepro 210x160x3 uint8 frame into 6400 (80x80x1) 2D float vector
        I = I[35:195]  # crop
        I = I[::2, ::2, 0]  # downsample by factor of 2
        I[I == 144] = 0  # erase background (background type 1)
        I[I == 109] = 0  # erase background (background type 2)
        I[I != 0] = 1  # everything else (paddles, ball) just set to 1
        I= np.reshape(I, (80, 80, 1))
        return I

    def discount_rewards(self, r, gamma=0.99):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, len(r))):
            if r[t] != 0: running_add = 0 # reset the sum
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        #normalize
        discounted_r -= np.mean(discounted_r)
        discounted_r /= np.std(discounted_r)
        return discounted_r

    def remember(self, state, action, actions, reward):
        # assure states are preprocessed before keep in memory
        if (state.shape[0] != 80 and state.shape[1] != 80):
            state = self.preprocess(state)

        # store in memory the different states, actions, rewards...
        self.memory.append((state, action, actions, reward))

    def replay(self):
        # fit model from memory
        gamma = 0.99 # importance of the next reward

        # initialize
        list_x_batch, list_y_batch = [], []
        
        # get the list of rewards
        _, _, _, list_r_batch = zip(*self.memory)
        r_batch = self.discount_rewards(list_r_batch, gamma) #process rewards
                
        for i, (state, action, actions, _) in enumerate(self.memory):

            state = np.expand_dims(state, axis=0)
            r = r_batch[i]  #reward of ith step
            #print("in) a:{} as:{} r:{}".format(action, actions, r))
            actions[action] += actions[action]*r
            #print("out) a:{} as:{} r:{}".format(action, actions, r))

            # append
            list_x_batch.append(state)
            list_y_batch.append(actions)
                
        # clean
        self.memory = []

        # train the model
        x_batch = np.vstack(list_x_batch)
        y_batch = np.vstack(list_y_batch)

        # fitting
        self.model.fit(x_batch, y_batch, verbose=0)

    def act(self, state):
        # preprocess the sample
        state = self.preprocess(state)
        state = np.expand_dims(state, axis=0)

        # predict the action to do
        action_values = self.model.predict(state)
        #print("Prediction({}) from {}".format(np.argmax(action_values), action_values))

        return np.argmax(action_values), action_values[0]
