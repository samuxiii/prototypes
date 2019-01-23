import os
import gym
import random
import numpy as np
from time import sleep
from nn import get_nn
from karpathy import prepro, discount_rewards


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
training = True
x_train, y_train, rewards = [], [], []
reward_sum = 0

# main loop
for i in range(100000):

    # predict action
    x = prepro(observation)
    proba = model.predict(np.expand_dims(x, axis=1).T)

    action = UP_ACTION if np.random.uniform() < proba else DOWN_ACTION
    y = 1 if action == 2 else 0  # 0 and 1 are our labels

    # do one step
    observation, reward, done, info = env.step(action)
    rewards.append(reward)
    reward_sum += reward

    # log the input and label to train later
    if training:
        x_train.append(x)
        y_train.append(y)
    else:
        env.render()
        sleep(0.03)

    if done:
        if training:
            model.fit(x=np.vstack(x_train), y=np.vstack(y_train), epochs=1, sample_weight=discount_rewards(rewards, 0.99))
            model.save_weights(h5file)
            x_train, y_train, rewards = [], [], []
            reward_sum = 0

        observation = env.reset()
