import gym
from gym import wrappers
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

# hyper parameters
EPISODES = 200  # number of episodes
EPS_START = 0.9  # e-greedy threshold start value
EPS_END = 0.05  # e-greedy threshold end value
EPS_DECAY = 2000  # e-greedy threshold decay
GAMMA = 0.99  # Q-learning discount factor
LR = 0.001  # NN optimizer learning rate
HIDDEN_LAYER = 256  # NN hidden layer size
BATCH_SIZE = 64  # Q-learning batch size

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Network(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(100800, HIDDEN_LAYER)
        self.l2 = nn.Linear(HIDDEN_LAYER, 6)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


env = gym.make('Pong-v0').unwrapped

model = Network()
if use_cuda:
    model.cuda()
memory = ReplayMemory(1000000)
optimizer = optim.Adam(model.parameters(), LR)
steps_done = 0
episode_durations = []

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(episode_durations)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    # with torch.no_grad():
    #     # print(model(state).data)
    #     return model(state).data.max(1)[1].view(1, 1)

    if sample > eps_threshold:
        with torch.no_grad():
            # print(model(state).data)
            return model(state).data.max(1)[1].view(1, 1)
    else:
        return LongTensor([[random.randrange(6)]])


def learn():
    if len(memory) < BATCH_SIZE:
        return

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(BATCH_SIZE)

    batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)

    batch_state = Variable(torch.cat(batch_state))
    batch_action = Variable(torch.cat(batch_action))
    batch_reward = Variable(torch.cat(batch_reward))
    batch_next_state = Variable(torch.cat(batch_next_state))

    # current Q values are estimated by NN for all actions
    current_q_values = model(batch_state).gather(1, batch_action)
    # expected Q values are estimated from actions which gives maximum Q value
    max_next_q_values = model(batch_next_state).detach().max(1)[0]
    expected_q_values = batch_reward + (GAMMA * max_next_q_values)

    # loss is measured from error between current and newly expected Q values

    loss = F.smooth_l1_loss(current_q_values.flatten(), expected_q_values.flatten())

    # backpropagation of loss to NN
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


for e in range(EPISODES):
    # run_episode(e, env)
    state = env.reset()
    steps = 0

    testIndex = 0

    while True:
        env.render()
        action = select_action(FloatTensor([state.reshape(100800)]))
        next_state, reward, done, _ = env.step(action[0, 0].item())

        print (action[0, 0].item(), reward)

        # negative reward when attempt ends
        if done:
            reward = -1

        memory.push((FloatTensor([state.reshape(100800)]),
                     action,  # action is already a tensor
                     FloatTensor([next_state.reshape(100800)]),
                     FloatTensor([reward])))

        learn()

        state = next_state
        steps += 1


        if done or steps >= 100:
            # print("Episode {0} finished after {1} steps".format(e, steps))
            episode_durations.append(steps)
            # plot_durations()
            break

# print('Complete')
# env.render(close=True)
# env.close()
# plt.ioff()
# plt.show()