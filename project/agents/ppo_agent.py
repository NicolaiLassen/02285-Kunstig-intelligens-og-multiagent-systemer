from copy import deepcopy
from typing import Dict

import torch
from torch import nn

from utils.buffer import MemBuffer
from utils.misc import normalize_dist


class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
                .log_prob(actions.squeeze(-1))
                .view(actions.size(0), -1)
                .sum(-1)
                .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)


class PPOAgent(object):
    def __init__(self,
                 actor: nn.Module,
                 critic: nn.Module,
                 eval: bool = False,
                 gamma=0.9,
                 eta=0.5,
                 beta=0.8,
                 eps_c=0.2,
                 loss_entropy_c=0.01,
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
            self.critic.eval()
            self.actor.eval()

        self.critic = critic
        self.actor = actor
        self.actor_old = deepcopy(actor)
        self.dist = Categorical(128, 29).cuda()  # 29 actions

        self.gamma = gamma
        self.eta = eta
        self.beta = beta

        self.eps_c = eps_c
        self.loss_entropy_c = loss_entropy_c

    def __repr__(self):
        return str({'actor': self.actor,
                    'critic': self.critic})

    def act(self, state):
        action_logs_prob = self.actor_old(state[0].unsqueeze(0).unsqueeze(0),
                                          state[1].unsqueeze(0).unsqueeze(0),
                                          state[2].unsqueeze(0).unsqueeze(0),
                                          state[3].unsqueeze(0))
        action_dist = self.dist(action_logs_prob)
        action = action_dist.mode()
        action_dist_log_prob = action_dist.log_prob(action)
        return action.detach().item(), action_dist.probs.detach(), action_dist_log_prob.detach()

    def evaluate(self, transitions: MemBuffer):
        action_logs_prob = self.actor(transitions.map.unsqueeze(1),
                                      transitions.map_goal.unsqueeze(1),
                                      transitions.map_color.unsqueeze(1),
                                      transitions.agent_pos)
        dist = self.dist(action_logs_prob)
        action_log_prob = dist.log_prob(transitions.actions)
        state_values = self.critic(transitions.map,
                                   transitions.map_goal)
        return action_log_prob, state_values.squeeze(1), dist.entropy()

    def loss(self, transitions: MemBuffer):
        action_log_probs, state_values, entropy = self.evaluate(transitions)
        d_r = self.__discounted_rewards(transitions)
        A_T = self.__advantages(d_r, state_values)
        R_T = normalize_dist(A_T)

        actor_loss = - self.__clipped_surrogate_objective(transitions, action_log_probs, R_T)  # L^CLIP

        critic_loss = (0.5 * torch.pow(state_values - d_r, 2)).mean()  # E # c1 L^VF

        entropy_bonus = entropy * self.loss_entropy_c  # c2 S[]

        # Gradient ascent -(actor_loss - critic_loss + entropy_bonus)
        return (actor_loss + critic_loss - entropy_bonus).mean()

    def __discounted_rewards(self, transitions: MemBuffer):
        discounted_rewards = torch.zeros(len(transitions.rewards))
        running_reward = 0
        t = 0
        for r, d in zip(reversed(transitions.rewards), reversed(transitions.done)):
            running_reward = r + (running_reward * self.gamma) * (1. - d)  # Zero out done states
            discounted_rewards[-t] = running_reward
            t += 1
        return discounted_rewards.float().cuda()

    def __advantages(self, discounted_rewards, state_values):
        advantages = torch.zeros(len(discounted_rewards))
        T = len(discounted_rewards) - 1
        last_state_value = state_values[T]
        t = 0
        for discounted_reward in discounted_rewards:
            advantages[t] = discounted_reward - state_values[t] + last_state_value * (
                    self.gamma ** (T - t))
            t += 1
        return advantages.float().cuda().detach()

    def __clipped_surrogate_objective(self, transitions: MemBuffer, action_log_probs, A_T):
        r_T_theta = torch.exp(action_log_probs - transitions.action_log_prob)
        r_T_c_theta = torch.clamp(r_T_theta, min=1 - self.eps_c, max=1 + self.eps_c)
        return torch.min(r_T_theta * A_T, r_T_c_theta * A_T).mean()  # E

    def eval(self):
        self.actor.eval()
        self.critic.eval()
        self.actor_old.eval()
        self.evaluating = True

    def train(self):
        self.actor.train()
        self.critic.train()
        self.evaluating = False

    def get_params(self):
        return {'actor': self.actor.state_dict(),
                'critic': self.critic.state_dict(),
                'cat': self.critic.state_dict()}

    def restore(self, params: Dict):
        self.actor.load_state_dict(params['actor'])
        self.actor_old.load_state_dict(params['actor'])
        self.critic.load_state_dict(params['critic'])
        self.dist.load_state_dict(params['cat'])
