import random
import numpy as np
from SumTree import SumTree


class Memory:
    e = 0.01  # epsilon which is a small positive constant that prevents edge-case of transitions not being revisited
    # once their error is zero
    alpha = 0.6
    beta = 0.4

    def __init__(self, memory_size, num_state):
        self.sumtree = SumTree(memory_size, num_state)
        self.memory_size = memory_size
        self.num_state = num_state

    def _get_priority(self, td_error):
        return (np.abs(td_error) + self.e) ** self.alpha  # 公式1中的p^alpha

    def add(self, td_error, transition):
        priority = self._get_priority(td_error)
        self.sumtree.add(priority, transition)

    def sample(self, batch_size, i_episode, episode_max):
        batch = np.zeros((batch_size, self.sumtree.memory.shape[1]))
        indexs = np.zeros((batch_size, 1), dtype=int)
        priorities = np.zeros((batch_size, 1))
        segment = self.sumtree.total() / batch_size
        beta = np.min([1., self.beta + (1 - self.beta) * i_episode / episode_max])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (index, priority, transition) = self.sumtree.get(s)
            indexs[i, 0] = index
            priorities[i, 0] = priority
            batch[i:i + 1, :] = transition

        P = priorities / self.sumtree.total()
        omega = np.power(self.sumtree.n_entries * P, -beta)
        omega = omega / np.max(omega)

        return batch, indexs, omega

    def update(self, index, td_error):
        priority = self._get_priority(td_error)
        self.sumtree.update(index, priority)
