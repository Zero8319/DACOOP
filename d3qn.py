import torch
import torch.nn as nn
import numpy as np
import environment
import torch.nn.functional as F
import time
from matplotlib import pyplot as plt
import random
from prioritized_memory import Memory

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Hyper Parameters
episode_max = 8000
batch_size = 128
lr = 3e-4
epsilon_origin = 1
epsilon_decrement = 1 / 4000
gamma = 0.99
target_replace_iter = 200
memory_size = int(1e6)
env = environment.environment(gamma)

num_action = env.num_action
num_state = env.num_state
num_agent = env.num_agent

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(2, 128)
        self.layer2 = nn.Linear(128 + 5, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(128, 64)
        self.layer5 = nn.Linear(64, num_action)
        self.layer6 = nn.Linear(64, 1)

    def forward(self, x):
        state_loc = x[:, :5]
        state_ij = x[:, 5:]
        x = state_ij.reshape(-1, 2)
        x = torch.relu(self.layer1(x))
        feature = torch.vstack((x[0::2, :].reshape(1, -1), x[1::2, :].reshape(1, -1)))
        mean_feature = torch.mean(feature, dim=0, keepdim=True).reshape(-1, x.shape[1])
        mean_feature = torch.hstack((state_loc, mean_feature))
        x = torch.relu(self.layer2(mean_feature))
        advantage = torch.relu(self.layer3(x))
        state_value = torch.relu(self.layer4(x))
        advantage = self.layer5(advantage)
        state_value = self.layer6(state_value)
        action_value = state_value + advantage - torch.mean(advantage, dim=1, keepdim=True)

        return action_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_counter = 0
        self.memory = Memory(memory_size, env.num_state)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss(reduction='none')
        self.max_td_error = 0.

    def choose_action(self, state, epsilon):
        state = torch.tensor(np.ravel(state), dtype=torch.float32, device=device).view(1, -1)
        if np.random.uniform() > epsilon:
            actions_value = self.eval_net(state)
            action = torch.max(actions_value, 1)[1].to('cpu').data.numpy().item()

        else:
            action = np.random.randint(0, num_action)
        return action

    def store_transition(self, state, action, reward, state_next, done):
        transition = np.hstack(
            (np.ravel(state), np.ravel(action), np.ravel(reward), np.ravel(state_next), np.ravel(done)))
        self.memory.add(self.max_td_error, transition)

    def learn(self, i_episode):
        if self.learn_step_counter % target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        b_memory, indexs, omega = self.memory.sample(batch_size, i_episode, episode_max)
        b_state = torch.tensor(b_memory[:, :num_state], dtype=torch.float32, device=device)
        b_action = torch.tensor(b_memory[:, num_state:num_state + 1], dtype=torch.int64, device=device)
        b_reward = torch.tensor(b_memory[:, num_state + 1:num_state + 2], dtype=torch.float32, device=device)
        b_state_next = torch.tensor(b_memory[:, num_state + 2:num_state * 2 + 2], dtype=torch.float32, device=device)
        b_done = torch.tensor(b_memory[:, num_state * 2 + 2:num_state * 2 + 3], dtype=torch.float32, device=device)

        q_eval = self.eval_net(b_state).gather(1, b_action)
        q_next_targetnet = self.target_net(b_state_next)
        q_next_evalnet = self.eval_net(b_state_next)
        q_target = b_reward + gamma * torch.abs(1 - b_done) * q_next_targetnet.gather(1,
                                                                                      torch.argmax(q_next_evalnet,
                                                                                                   axis=1,
                                                                                                   keepdim=True))
        td_errors = (q_target - q_eval).to('cpu').detach().numpy().reshape((-1, 1))
        self.max_td_error = max(np.max(np.abs(td_errors)), self.max_td_error)
        for i in range(batch_size):
            index = indexs[i, 0]
            td_error = td_errors[i, 0]
            self.memory.update(index, td_error)

        loss = (self.loss_func(q_eval, q_target.detach()) * torch.FloatTensor(omega).to(device).detach()).mean()
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.eval_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


class RunningStat:
    def __init__(self, shape):
        self.n = 0
        self.mean = np.zeros(shape)
        self.s = np.zeros(shape)
        self.std = np.zeros(shape)

    def push(self, x):
        self.n += 1
        if self.n == 1:
            self.mean = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.s = self.s + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.s / (self.n - 1) if self.n > 1 else np.square(self.mean))


dqn = DQN()
running_stat = RunningStat((1,))
device = torch.device('cuda')
dqn.eval_net.to(device)
dqn.target_net.to(device)

i_episode = 0

episode_return_total = np.zeros(0)

while True:
    state = env.reset()
    last_done = np.array([[0., 0, 0]])
    episode_return = 0

    while True:
        # env.render()
        action = np.zeros((1, num_agent))
        for i in range(num_agent):
            temp = state[:, i:i + 1].reshape(1, -1)
            action_temp = dqn.choose_action(temp, max(epsilon_origin - epsilon_decrement * i_episode, 0.01))
            action[0, i] = action_temp
        state_next, reward, done = env.step(action)

        for i in range(num_agent):
            if not np.ravel(last_done)[i]:
                episode_return += np.ravel(reward)[i]
                running_stat.push(reward[:, i])
                reward[0, i] = np.clip(reward[0, i] / (running_stat.std + 1e-8), -10, 10)
                dqn.store_transition(state[:, i:i + 1], action[:, i:i + 1], reward[:, i:i + 1], state_next[:, i:i + 1],
                                     done[:, i:i + 1])

        if np.all(done):
            if dqn.memory.sumtree.n_entries == memory_size:
                for _ in range(1000):
                    dqn.learn(i_episode)
                i_episode += 1
            break
        state = state_next
        last_done = done

    if dqn.memory.sumtree.n_entries < memory_size:
        temp = "collecting experiences: " + str(dqn.memory.sumtree.n_entries) + ' / ' + str(memory_size)
        print(temp)

    if dqn.memory.sumtree.n_entries == memory_size:
        episode_return_total = np.hstack((episode_return_total, episode_return))
        plt.figure(1)
        plt.cla()
        plt.plot(episode_return_total)
        plt.show(block=False)
        plt.pause(0.01)
        print('i_episode: ', i_episode, 'episode_return: ', round(episode_return, 2))

    if i_episode % 100 == 0:
        net = dqn.eval_net
        string = str(i_episode) + '.pt'
        torch.save(net.state_dict(), string)
        string = str(i_episode) + '.txt'
        np.savetxt(string, episode_return_total)
    if i_episode == episode_max:
        break

