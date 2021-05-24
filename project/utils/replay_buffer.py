import numpy as np
from torch import Tensor
from torch.autograd import Variable


class ReplayBuffer:

    def __init__(self, capacity=1e6, batch_size=100, width=50, height=50, action_space_n=29, max_agents_n=10):
        self.capacity = capacity
        self.batch_size = batch_size
        self.max_agents_n = max_agents_n

        self.cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        self.s = []
        self.a = []
        self.r = []
        self.s1 = []
        self.d = []

        self.f_t = 0
        self.t = 0
        # for i in range(max_agents_n):
        #     print()

    def append(self, s, a, r, s1, d):
        print()
        # if self.t > self.capacity:
        #     for agent_i in range(self.max_agents_n):
        #         print()
        #
        # for agent_i in range(self.max_agents_n):
        #     print()

        self.t += 1

    def sample(self):
        indexs = np.random.choice(np.arange(self.f_t), size=self.batch_size, replace=False)
        print(indexs)
        return ([self.cast(self.s[i][indexs]) for i in range(self.max_agents_n)],
                [self.cast(self.a[i][indexs]) for i in range(self.max_agents_n)],
                [self.cast(self.r[i][indexs]) for i in range(self.max_agents_n)],
                [self.cast(self.s1[i][indexs]) for i in range(self.max_agents_n)],
                [self.cast(self.d[i][indexs]) for i in range(self.max_agents_n)])

    def __len__(self):
        return self.t
