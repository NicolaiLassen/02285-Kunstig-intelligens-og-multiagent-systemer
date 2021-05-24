from copy import deepcopy
from typing import Dict, Union, Any, Tuple

import torch
from torch import nn, Tensor
from torch.optim import Adam

from agents.ppo_agent import PPOAgent
from environment.env_wrapper import MultiAgentEnvWrapper
from utils.buffer import MultiAgentMemBuffer


# https://arxiv.org/pdf/2103.01955.pdf
# The Surprising Effectiveness of MAPPO in Cooperative, Multi-Agent Games
class MAPPOTrainer(object):
    replay_buffer: MultiAgentMemBuffer = None

    def __init__(self,
                 actor: nn.Module,
                 critic: nn.Module,
                 env: MultiAgentEnvWrapper = None,
                 actor_lr: float = 0.0005,
                 critic_lr: float = 0.001,
                 eval=False,
                 gamma=0.9,
                 eta=0.5,
                 beta=0.8,
                 eps_c=0.2,
                 loss_entropy_c=0.01,
                 ):
        self.env = env

        self.optimizer = Adam([
            {'params': actor.parameters(), 'lr': actor_lr},
            {'params': critic.parameters(), 'lr': critic_lr}
        ])

        agent = PPOAgent(actor.cuda(),
                         critic.cuda(),
                         gamma=gamma,
                         eta=eta,
                         beta=beta,
                         eps_c=eps_c,
                         loss_entropy_c=loss_entropy_c,
                         eval=eval)

        self.agents: Dict[int, PPOAgent] = {i: deepcopy(agent) for i in range(self.env.max_agents_n)}

        self.action_space_n = self.env.action_space_n

    def act(self, agent_states: Dict[Union[str, int], Any]) -> Tuple[
        Dict[Union[int], Tensor], Dict[Union[int], Tensor], Dict[Union[int], Tensor]]:
        actions = {}
        probs = {}
        log_probs = {}
        for a_key in agent_states:
            state = agent_states[a_key]
            action, prob, log_prob = self.agents[a_key].act(state)
            actions[a_key] = action
            probs[a_key] = prob
            log_probs[a_key] = log_prob
        return actions, probs, log_probs

    def train(self, step_max: int, batch=2000, horizon: int = 1000, save_every=200000, path: str = "./ckpt"):
        t = 0
        s1 = self.env.reset()
        self.replay_buffer = MultiAgentMemBuffer(max_length=batch)
        while t < step_max:
            for i in range(batch):
                if t % save_every == 0:
                    self.save("{}/agent_{}.ckpt".format(path, t))

                t += 1
                s = s1
                actions, probs, log_probs = self.act(s)
                s1, r, d, _ = self.env.step(actions)
                self.replay_buffer.append(s, actions, r, s1, probs, log_probs, d)
                if t % horizon == 0 or d:
                    s1 = self.env.reset()
            self.update()

    def update(self):
        torch.cuda.empty_cache()

        loss = torch.tensor(0).float().cuda()
        for a_key in range(len(self.agents)):
            transition_buffer = self.replay_buffer.agent_buffer(a_key)
            loss += self.agents[a_key].loss(transition_buffer)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.replay_buffer.clear()

    def save(self, path: str):
        save_dict = {'agent_params': [self.agents[a_key].get_params() for a_key in self.agents]}
        torch.save(save_dict, path)

    def eval(self):
        for a_key in self.agents:
            self.agents[a_key].eval()

    def restore(self, path: str):
        save_dict = torch.load(path)
        for a_key, params in zip(self.agents, save_dict['agent_params']):
            self.agents[a_key].restore(params)
