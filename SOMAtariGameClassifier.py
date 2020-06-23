from minisom import MiniSom
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gym
from sklearn.datasets import load_breast_cancer

atariGames = ['Pong-v0', 'Breakout-v0', 'MsPacman-v0', 'SpaceInvaders-v0', 'Seaquest-v0']

def getStateSample(games):
    data = []
    target = []
    index = 0
    for i in games:
        env = gym.make(i)
        observation = env.reset()
        for t in range(1000):
            observation, reward, done, info = env.step(env.action_space.sample())
            data.append(observation.reshape(100800))
            target.append(index)
        env.close()
        index += 1
    return np.array(data), np.array(target)

data, target = getStateSample(atariGames)

print (data)
print (target)


som_grid_rows = 30
som_grid_column = 20
iterations = 500
sigma = 1
learning_rate = 0.5

som = MiniSom(x=som_grid_rows,
              y=som_grid_column,
              input_len=data.shape[1],
              sigma=sigma,
              learning_rate=learning_rate)

som.random_weights_init(data)

start_time = time.time()
som.train_random(data, iterations)
elapsed_time = time.time() - start_time
print (elapsed_time, " second")

from pylab import plot, axis, show, pcolor, colorbar, bone
bone()
pcolor(som.distance_map().T)
colorbar()

markers = ['o', 's', 'D', 'v', 'X']
colors = ['r', 'g', 'b', 'c', 'm']
for cnt, xx in enumerate (data):
    w = som.winner(xx)
    plot(w[0] + .5, w[1] + .5, markers[target[cnt]], markerfacecolor='None', markeredgecolor=colors[target[cnt]], markersize=12, markeredgewidth=2)
axis([0, som._weights.shape[0], 0, som._weights.shape[1]])
show()


