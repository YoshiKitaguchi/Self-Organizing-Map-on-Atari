from minisom import MiniSom

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gym

atariGames = ['Pong-v0', 'Breakout-v0', 'MsPacman-v0', 'SpaceInvaders-v0', 'Seaquest-v0']

def getStateSample(games):
    observations = []
    for i in games:
        # print(i)
        env = gym.make(i)
        observation = env.reset()
        for t in range(3):
            # print('loop: ', t)
            observation, reward, done, info = env.step(env.action_space.sample())
            # print(observation.reshape(100800))
            observations.append(observation.reshape(100800))
        env.close()
        print()
    return np.array(observations)

print(getStateSample(atariGames))

