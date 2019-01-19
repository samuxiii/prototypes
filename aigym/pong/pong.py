import gym
import random
import numpy as np
from time import sleep
from nn import get_nn

def prepro(I):
# prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

# code for the two only actions in Pong
UP_ACTION = 2
DOWN_ACTION = 3

# initializing our environment
env = gym.make("Pong-v0")

# beginning of an episode
observation = env.reset()

# get model
model = get_nn()


x_train, y_train = [], []
# main loop
for i in range(10000):
    action = random.randint(UP_ACTION, DOWN_ACTION)

    env.render()
    observation, reward, done, info = env.step(action)

    x = prepro(observation)
    proba = model.predict(np.expand_dims(x, axis=1).T)

    action = UP_ACTION if np.random.uniform() < proba else DOWN_ACTION
    y = 1 if action == 2 else 0  # 0 and 1 are our labels

    # log the input and label to train later
    x_train.append(x)
    y_train.append(y)

    if done:
        model.fit(x=np.vstack(x_train), y=np.vstack(y_train))
        x_train, y_train = [], []
        env.reset()

    ##
    sleep(0.03)
    print(sum(prepro(observation)))
