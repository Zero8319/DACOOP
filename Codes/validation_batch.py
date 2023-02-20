import torch
import numpy as np
import torch.nn as nn
import environment_validation
import random
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
randomseed = 1234
gamma = 0.99
torch.manual_seed(randomseed)
np.random.seed(randomseed)
random.seed(randomseed)
env = environment_validation.environment(gamma)
num_state = env.num_state
num_action = env.num_action


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
        mean_feature = torch.nanquantile(feature, 0.5, dim=0).reshape(-1, x.shape[1])
        mean_feature = torch.hstack((state_loc, mean_feature))
        x = torch.relu(self.layer2(mean_feature))
        advantage = torch.relu(self.layer3(x))
        state_value = torch.relu(self.layer4(x))
        advantage = self.layer5(advantage)
        state_value = self.layer6(state_value)
        action_value = state_value + advantage - torch.mean(advantage, dim=1, keepdim=True)

        return action_value


filenames = ['8000']
net = Net()
for file in filenames:
    net.load_state_dict(torch.load(file + '.pt'))

    return_per_episode = np.zeros((0, 1))
    t_per_episode = np.zeros((0, 1))
    action_value_record = list(
        [np.zeros((0, env.num_action)), np.zeros((0, env.num_action)), np.zeros((0, env.num_action))])
    for episode_counter in range(1000):
        j = 0
        print(episode_counter)
        state = env.reset()
        last_done = np.array([[0., 0, 0]])
        episode_return = np.array([[0.]])
        while True:
            env.render()
            # plt.savefig(str(j))
            j += 1
            action = np.zeros((1, 0))
            for i in range(env.num_agent):
                temp = state[:, i:i + 1].reshape(1, -1)
                temp = net(torch.tensor(np.ravel(temp), dtype=torch.float32).view(1, -1))
                # action_value_record[i] = np.vstack((action_value_record[i], temp.data.numpy()))
                action_temp = torch.max(temp, 1)[1].data.numpy()
                action = np.hstack((action, np.array(action_temp, ndmin=2)))
            # take action
            state_next, reward, done = env.step(action)
            for i in range(env.num_agent):
                if not np.ravel(last_done)[i]:
                    episode_return += np.ravel(reward)[i]
            temp1 = done == 1
            temp2 = done == 2
            temp = np.vstack((temp1, temp2))
            if np.all(np.any(temp, axis=0, keepdims=True)):
                return_per_episode = np.vstack((return_per_episode, episode_return))
                t_per_episode = np.vstack((t_per_episode, np.array(env.t, ndmin=2)))
                break
            state = state_next
            last_done = done
    file_name = file + '_return.txt'
    np.savetxt(file_name, return_per_episode)
    file_name = file + '_t.txt'
    np.savetxt(file_name, t_per_episode)
    # np.savetxt('action_value_record0.txt', action_value_record[0])
    # np.savetxt('action_value_record1.txt', action_value_record[1])
    # np.savetxt('action_value_record2.txt', action_value_record[2])

###### plot ########
# success = list()
# mean_t = list()
# reward = list()
# for i in range(21):
#     filename = str(i * 500) + "_return.txt"
#     temp = np.loadtxt(filename)
#     reward.append(np.mean(temp))
#     filename = str(i * 500) + "_t.txt"
#     temp = np.loadtxt(filename)
#     mean_t.append(np.mean(temp))
#     success.append(np.sum(temp < 1000))
# plt.figure(1)
# plt.plot(reward)
# plt.ylabel("reward")
# plt.figure(2)
# plt.plot(success)
# plt.ylabel("success")
# plt.figure(3)
# plt.plot(mean_t)
# plt.ylabel("mean_t")
