from copy import deepcopy
from typing import Callable, Dict, Union

import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Optimizer, Adam

from utils.misc import logit_onehot


class DDPG(object):
    def __init__(self,
                 actor: nn.Module,
                 critic: nn.Module,
                 optimizer: Union[Optimizer, Callable] = Adam,
                 actor_lr: float = 0.0005,
                 critic_lr: float = 0.001,
                 eval: bool = False
                 ):
        """
        Inputs:
            actor (nn.Module): The policy model nn.Module
            critic (nn.Module): The state estimation model nn.Module
            optimizer (U nn.Optimizer ∩ Callable): optimizer used for AC
            eps (ɛ): If policy should act after ɛ-greedy exploration
            eval: If the agent should explore og eval its model
        """

        self.evaluating = eval
        if eval:
            self.Q.eval()
            self.pi.eval()

        self.Q = critic  # Value estimator
        self.target_Q = deepcopy(critic)

        self.pi = actor  # Policy φ
        self.target_pi = deepcopy(actor)

        self.Q_optimizer = optimizer(self.Q.parameters(), lr=critic_lr)
        self.pi_optimizer = optimizer(self.pi.parameters(), lr=actor_lr)

    def __repr__(self):
        return str({'actor': self.pi,
                    'critic': self.Q,
                    'target_actor': self.target_pi,
                    'target_critic': self.target_Q,
                    'actor_optimizer': self.pi_optimizer,
                    'critic_optimizer': self.Q_optimizer})

    def act(self, state, explore=True):
        logits: Tensor = self.pi(state)
        if explore:
            action = F.gumbel_softmax(logits, hard=True)
        else:
            action = logit_onehot(logits)
        return action

    def eval(self):
        self.pi.eval()
        self.Q.eval()
        self.evaluating = True

    def train(self):
        self.pi.train()
        self.Q.train()
        self.evaluating = False

    def get_params(self):
        return {'actor': self.pi.state_dict(),
                'critic': self.Q.state_dict(),
                'target_actor': self.target_pi.state_dict(),
                'target_critic': self.target_Q.state_dict(),
                'actor_optimizer': self.pi_optimizer.state_dict(),
                'critic_optimizer': self.Q_optimizer.state_dict()}

    def restore(self, params: Dict):
        self.pi.load_state_dict(params['actor'])
        self.Q.load_state_dict(params['critic'])
        self.target_pi.load_state_dict(params['target_actor'])
        self.target_Q.load_state_dict(params['target_critic'])
        self.pi_optimizer.load_state_dict(params['actor_optimizer'])
        self.Q_optimizer.load_state_dict(params['critic_optimizer'])
