from typing import Dict, List

from ray.rllib.train import torch


class MemBuffer:
    map = None
    map_goal = None
    map_color = None
    agent_pos = None

    next_states = None
    rewards = None
    actions = None
    action_probs = None
    action_log_prob = None
    done = None
    t = 0

    def __init__(self, max_length: int = int(500), width=50, height=50, action_space_n=29, max_agents_n=10):
        self.max_length = max_length
        self.action_space_n = action_space_n
        self.max_agents_n = max_agents_n
        self.width = width
        self.height = height
        self.agent_buffers = []
        self.clear()

    def append(self, s, a, r, s1, probs, log_probs, d: bool):
        self.map[self.t] = s[0]
        self.map_goal[self.t] = s[1]
        self.map_color[self.t] = s[2]
        self.agent_pos[self.t] = s[3]

        self.actions[self.t] = a
        self.rewards[self.t] = r
        self.action_probs[self.t] = probs
        self.action_log_prob[self.t] = log_probs
        self.done[self.t] = d
        self.t += 1

    def clear(self):
        self.t = 0
        self.map = torch.zeros(self.max_length, self.height, self.width, dtype=torch.float).cuda()
        self.map_goal = torch.zeros(self.max_length, self.height, self.width, dtype=torch.float).cuda()
        self.map_color = torch.zeros(self.max_length, self.height, self.width, dtype=torch.float).cuda()
        self.agent_pos = torch.zeros(self.max_length, 2, dtype=torch.float).cuda()

        self.actions = torch.zeros(self.max_length, dtype=torch.long).cuda()
        self.rewards = torch.zeros(self.max_length, dtype=torch.float).cuda()
        self.action_probs = torch.zeros(self.max_length, self.action_space_n, dtype=torch.float).cuda()
        self.action_log_prob = torch.zeros(self.max_length, dtype=torch.float).cuda()
        self.done = torch.zeros(self.max_length, dtype=torch.int)

        # self.next_states = torch.zeros(self.max_length, self.height, self.width, dtype=torch.float).cuda()


class MultiAgentMemBuffer:
    t = 0

    def __init__(self, max_length: int, width=50, height=50, action_space_n=29, max_agents_n=10):
        self.max_length = max_length
        self.action_space_n = action_space_n
        self.max_agents_n = max_agents_n
        self.width = width
        self.height = height
        self.agent_buffers: List[MemBuffer] = []

        for i in range(self.max_agents_n):
            self.agent_buffers.append(MemBuffer(max_length=max_length))

    def append(self, s: Dict, a: Dict, r: Dict, s1: Dict, probs: Dict, log_probs: Dict, d: bool):
        for a_key in a:
            self.agent_buffers[a_key].append(s[a_key],
                                             a[a_key],
                                             r[a_key],
                                             s1[a_key],
                                             probs[a_key],
                                             log_probs[a_key],
                                             d)
        self.t += 1

    def agent_buffer(self, i) -> MemBuffer:
        return self.agent_buffers[i]

    def clear(self):
        self.t = 0
        for i in range(self.max_agents_n):
            self.agent_buffers[i].clear()

    def __len__(self):
        return self.t
