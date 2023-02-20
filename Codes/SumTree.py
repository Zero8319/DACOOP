import numpy


class SumTree:

    def __init__(self, memory_size, num_state):
        self.memory_size = memory_size
        self.tree = numpy.zeros((2 * memory_size - 1, 1))
        self.memory = numpy.zeros((memory_size, num_state * 2 + 2 + 1))
        self.memory_counter = 0
        self.n_entries = 0

    def _propagate(self, index, change):  # idx为某一结点,change为某一结点priority的改变量
        parent = (index - 1) // 2
        self.tree[parent, 0] = self.tree[parent, 0] + change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, index, s):  # idx为某一结点，s为采样值
        left = 2 * index + 1
        right = left + 1
        if left >= self.tree.size:
            return index
        if s <= self.tree[left, 0]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left, 0])

    def total(self):
        return self.tree[0, 0]

    def add(self, priority, transition):
        index = self.memory_counter + self.memory_size - 1
        self.memory[self.memory_counter:self.memory_counter + 1, :] = transition
        self.update(index, priority)

        self.memory_counter += 1
        if self.memory_counter >= self.memory_size:
            self.memory_counter = 0

        if self.n_entries < self.memory_size:
            self.n_entries += 1

    # update priority
    def update(self, index, priority):
        change = priority - self.tree[index, 0]  # change为第i个结点的priority改变量
        self.tree[index, 0] = priority
        self._propagate(index, change)  # 更新整个sumtree中的值

    # get priority and sample
    def get(self, s):  # s为采样值
        index = self._retrieve(0, s)
        dataIdx = index - self.memory_size + 1

        return (index, self.tree[index, 0], self.memory[dataIdx:dataIdx + 1, :])
