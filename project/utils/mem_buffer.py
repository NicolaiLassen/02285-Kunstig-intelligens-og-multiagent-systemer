import torch


# Takes at max 9 client
# zero out clients that are not in action

class AgentMemBuffer:
    t = 0

    def __init__(self, max_time, width=50, height=50, action_space_n=29, max_client=9):
        self.max_length = max_time
        self.width = width
        self.height = height
        self.action_space_n = action_space_n
        self.max_client = max_client

        self.map_states = torch.zeros(self.max_length, self.height, self.width, dtype=torch.float)
        self.map_next_states = torch.zeros(self.max_length, self.height, self.width, dtype=torch.float)

        # max train length x number of client in state, grid dirs xy
        self.agent_states = torch.zeros(self.max_length, max_client, 2, dtype=torch.float)

        self.rewards = torch.zeros(self.max_length, dtype=torch.float)
        self.actions = torch.zeros(self.max_length, dtype=torch.long)
        self.action_probs = torch.zeros(self.max_length, action_space_n, dtype=torch.float)
        self.action_log_prob = torch.zeros(self.max_length, dtype=torch.float)
        self.done = torch.zeros(self.max_length, dtype=torch.int)

    # State [map, agents]
    def set_next(self,
                 state,
                 next_states,
                 reward,
                 action,
                 action_probs,
                 action_log_prob,
                 done):
        if self.max_length == self.t:
            print("DON'T JUST TAKE ALL OF MY SPACE, YOU SON OF A GUN!")
            return

        self.map_states[self.t] = state[0]
        self.map_next_states[self.t][:len(next_states[0])] = next_states[0]
        self.agent_states[self.t] = state[1]
        self.rewards[self.t] = float(reward)
        self.actions[self.t] = action
        self.action_probs[self.t] = action_probs
        self.action_log_prob[self.t] = action_log_prob
        self.done[self.t] = int(done)
        self.t += 1

    def clear(self):
        self.map_states = torch.zeros(self.max_length, self.height, self.width, dtype=torch.float)
        self.map_next_states = torch.zeros(self.max_length, self.height, self.width, dtype=torch.float)
        self.agent_states = torch.zeros(self.max_length, self.max_client, 2, dtype=torch.float)
        self.rewards = torch.zeros(self.max_length, dtype=torch.float)
        self.actions = torch.zeros(self.max_length, dtype=torch.long)
        self.action_probs = torch.zeros(self.max_length, self.action_space_n, dtype=torch.float)
        self.action_log_prob = torch.zeros(self.max_length, dtype=torch.float)
        self.done = torch.zeros(self.max_length, dtype=torch.int)
        self.t = 0
