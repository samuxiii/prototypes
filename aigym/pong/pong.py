import os
import gym
import random
import numpy as np
from time import sleep
from nn import get_nn
from kaparthy import prepro


# code for the two only actions in Pong
UP_ACTION = 2
DOWN_ACTION = 3

# initializing our environment
env = gym.make("Pong-v0")

# beginning of an episode
observation = env.reset()

# model weights
h5file = "weights.h5"

# get model
model = get_nn()
if os.path.exists(h5file):
    model.load_weights(h5file)


# training conf
train = True
x_train, y_train = [], []

# main loop
for i in range(10000):
    action = random.randint(UP_ACTION, DOWN_ACTION)

    if not train:
        env.render()

    observation, reward, done, info = env.step(action)

    x = prepro(observation)
    proba = model.predict(np.expand_dims(x, axis=1).T)

    action = UP_ACTION if np.random.uniform() < proba else DOWN_ACTION
    y = 1 if action == 2 else 0  # 0 and 1 are our labels

    # log the input and label to train later
    if train:
        x_train.append(x)
        y_train.append(y)

    if done:
        if train:
            model.fit(x=np.vstack(x_train), y=np.vstack(y_train))
            model.save_weights("weights.h5")
            x_train, y_train = [], []
        env.reset()

    ##
    if not train:
        sleep(0.03)