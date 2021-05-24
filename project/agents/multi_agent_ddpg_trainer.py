from typing import Dict, Union, Any

import torch
import torch.nn.functional as F
from torch import Tensor

from agents.ddpg_agent import DDPG
from environment.env_wrapper import MultiAgentEnvWrapper
# https://arxiv.org/pdf/1706.02275.pdf
# ICM, Curiosity-driven Exploration by Self-supervised Prediction
# ref: https://github.com/iffiX/machin/blob/d10727b52d981c898e31cdd20b48a3d972612bb6/machin/frame/algorithms/maddpg.py#L500
from utils.misc import soft_update, logit_onehot
from utils.replay_buffer import ReplayBuffer


class MADDPGTrainer(object):
    replay_buffer_d = []

    def __init__(self,
                 env: MultiAgentEnvWrapper,
                 agents: Dict[Union[str, int], DDPG] = None,
                 criterion=torch.nn.MSELoss(reduction="sum"),
                 gamma=0.95,
                 tau=0.01,
                 eval=False):

        self.env = env
        self.criterion = criterion
        self.gamma = gamma
        self.tau = tau

        if eval:
            for a_key in agents:
                agents[a_key].eval()

        self.agents: Dict[Union[str, int], DDPG] = agents
        self.pis = [self.agents[a_key].pi for a_key in self.agents]
        self.target_pis = [self.agents[a_key].target_pi for a_key in self.agents]

    def act(self, agent_states: Dict[Union[str, int], Any], explore=True) -> Dict[Union[str, int], Tensor]:
        actions = {}
        for a_key in agent_states:
            state = agent_states[a_key]
            action = self.agents[a_key].act(state, explore)
            actions[a_key] = action
        return actions

    def train(self, episodes_n, capacity=int(1e6), horizion: int = 1000):
        t = 0
        self.replay_buffer_d = ReplayBuffer(capacity=capacity)
        s1 = self.env.reset()
        while t < episodes_n:
            # game does not show much reward so pivot to test envs
            for i in range(horizion):
                s = s1
                # for each agent i, select action ai = µθi (oi) + Nt w.r.t. the current policy and exploration
                a = self.act(s)
                action_idxs = {a_key: a[a_key].argmax(1).item() for a_key in a}
                # Execute actions a = (a1, . . . , aN ) and observe reward r and new state x'
                s1, r, d, _ = self.env.step(action_idxs)
                # Store T (x, a, r, x') in replay buffer D
                self.replay_buffer_d.append(s, a, r, d, s1)
                t += 1
                # reset env if done state
                if d:
                    s1 = self.env.reset()

            s1 = self.env.reset()
            self.update()

    def update(self):
        for a_key in self.agents:
            # Sample a random minibatch of S samples
            sample = self.replay_buffer_d.sample()
            # Update critic by minimizing the loss L(phi)
            self.__update_Q(sample, a_key)
            # Update actor using the sampled policy gradient
            self.__update_pi(sample, a_key)
        for a_key in self.agents:
            agent = self.agents[a_key]
            # Soft tau update target network parameters for each agent i
            soft_update(agent.target_Q, agent.Q, self.tau)
            soft_update(agent.target_pi, agent.pi, self.tau)

    def __update_Q(self, sample, a_key):
        s, a, r, s1, d = sample
        c_agent = self.agents[a_key]

        c_agent.Q_optimizer.zero_grad()
        all_trgt_acs = [logit_onehot(pi(s)) for pi, s in zip(self.target_pis, s1)]
        trgt_vf_in = torch.cat((*s1, *all_trgt_acs), dim=1)

        target_value = (r[a_key].view(-1, 1) + self.gamma * c_agent.target_Q(trgt_vf_in) * (1 - d[a_key].view(-1, 1)))
        vf_in = torch.cat((*s, *a), dim=1)
        actual_value = c_agent.Q(vf_in)
        vf_loss = self.criterion(actual_value, target_value.detach())
        vf_loss.backward()

        torch.nn.utils.clip_grad_norm(c_agent.Q.parameters(), 0.5)
        c_agent.Q_optimizer.step()

    def __update_pi(self, sample, a_key):
        s, a, r, s1, d = sample
        c_agent = self.agents[a_key]

        c_agent.pi_optimizer.zero_grad()
        curr_pi_out = c_agent.pi(s[a_key])
        curr_pi_vf_in = F.gumbel_softmax(curr_pi_out, hard=True)
        all_pi_acs = []

        for i, pi, ob in zip(range(self.env.max_agents_n), self.pis, s):
            if i == a_key:
                all_pi_acs.append(curr_pi_vf_in)
            else:
                all_pi_acs.append(logit_onehot(pi(ob)))

        vf_in = torch.cat((*s, *all_pi_acs), dim=1)

        pi_loss = -c_agent.Q(vf_in).mean()
        pi_loss += (curr_pi_out ** 2).mean() * 1e-3  # const
        pi_loss.backward()

        torch.nn.utils.clip_grad_norm(c_agent.pi.parameters(), 0.5)
        c_agent.pi_optimizer.step()

    def save(self, path: str):
        save_dict = {'agent_params': [self.agents[a_key].get_params() for a_key in self.agents]}
        torch.save(save_dict, path)

    def restore(self, path: str):
        save_dict = torch.load(path)
        for a_key, params in zip(self.agents, save_dict['agent_params']):
            self.agents[a_key].restore(params)
