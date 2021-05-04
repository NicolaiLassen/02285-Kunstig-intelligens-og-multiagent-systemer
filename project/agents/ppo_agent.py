import copy

import torch.nn as nn
import torch.optim as optim

from agents.agent_base import BaseAgent
from models.curiosity import ICM
from utils import mem_buffer


# PPO Actor Critic
class PPOAgent(BaseAgent):
    mem_buffer: mem_buffer = None

    def __init__(self,
                 action_space_n: int,
                 actor: nn.Module,
                 critic: nn.Module,
                 optimizer: optim.Optimizer,
                 n_acc_gradient=20,
                 gamma=0.9,
                 eps_c=0.2,
                 n_max_Times_update=1):

        self.actor = actor
        self.actor_old = copy.deepcopy(actor)
        self.actor_old.load_state_dict(actor.state_dict())
        self.critic = critic
        self.optimizer = optimizer
        self.action_space_n = action_space_n
        # Curiosity
        self.ICM = ICM()
        # Hyper n
        self.n_acc_grad = n_acc_gradient
        self.n_max_Times_update = n_max_Times_update
        # Hyper c
        self.gamma = gamma
        self.eps_c = eps_c
        self.loss_entropy_c = 0.01
        self.intrinsic_curiosity_c = 0.9

    def act(self, state) -> int:
        return 0

    def train(self, max_time, max_time_steps):
        return
