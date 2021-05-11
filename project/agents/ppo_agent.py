import copy
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.distributions import Categorical

from agents.agent_base import BaseAgent
from environment.action import idxs_to_actions
from environment.env_wrapper import EnvWrapper
from models.curiosity import IntrinsicCuriosityModule
from models.policy_models import ActorPolicyModel
from utils import mem_buffer
# PPO Actor Critic
from utils.mem_buffer import AgentMemBuffer
# TODO: WE NEED TO BE DONE
# TODO: UPDATE WITH NEW RL IMC
# TODO: MEM BUFFER
# TODO: REWARD
# TODO: Optim code performance
# TODO: self.mem_buffer.states [map,agents], [map,colors,agents]
# TODO: CUDA for train
from utils.normalize_dist import normalize_dist


class PPOAgent(BaseAgent):
    mem_buffer: mem_buffer = None

    def __init__(self,
                 env: EnvWrapper,
                 actor: nn.Module,
                 critic: nn.Module,
                 optimizer: optim.Optimizer,
                 n_acc_gradient=10,
                 gamma=0.9,
                 lamda=0.5,
                 eta=0.5,
                 beta=0.8,
                 eps_c=0.2,
                 loss_entropy_c=0.01,
                 n_max_Times_update=1):
        self.env = env
        self.actor = actor
        self.actor_old = copy.deepcopy(actor)
        self.actor_old.load_state_dict(actor.state_dict())
        self.critic = critic
        self.optimizer = optimizer

        self.max_agents = 9  # HARDCODE 9

        self.action_space_n = env.action_space_n
        # Curiosity
        self.ICM = IntrinsicCuriosityModule(self.action_space_n)
        # Hyper n
        self.n_acc_grad = n_acc_gradient
        self.n_max_Times_update = n_max_Times_update
        # Hyper c
        self.gamma = gamma
        self.lamda = lamda
        self.eta = eta
        self.beta = beta

        self.eps_c = eps_c
        self.loss_entropy_c = loss_entropy_c

    def train(self, max_Time: int, max_Time_steps: int):
        self.mem_buffer = AgentMemBuffer(max_Time, action_space_n=self.action_space_n)
        t = 0

        while t < max_Time_steps:
            self.save_actor()
            s1 = self.env.reset()
            for ep_T in range(max_Time + 1):
                s = s1
                action_idxs, probs, log_prob = self.act(s)
                actions = idxs_to_actions(action_idxs)

                temp_step = self.env.step(actions)
                if temp_step is None:
                    continue

                t += 1
                s1, r, d = temp_step
                self.mem_buffer.set_next(s, s1, r, action_idxs, probs, log_prob, d)
                if t % max_Time == 0:
                    self.__update()

    def act(self, state) -> Tuple[Tensor, Tensor, Tensor]:
        state_map = normalize_dist(state[0])
        actions_logs_prob = self.actor_old(state_map.unsqueeze(0).unsqueeze(0),
                                           state[1].unsqueeze(0),
                                           self.env.mask)

        actions_dist = Categorical(actions_logs_prob)
        actions = actions_dist.sample()
        action_dist_log_prob = actions_dist.log_prob(actions)
        return actions.detach(), actions_dist.probs.detach(), action_dist_log_prob.detach()

    def save_actor(self):
        return
        # torch.save(self.actor_old.state_dict(), "encoder_actor.ckpt")

    def load_actor(self, path):
        self.actor.load_state_dict(torch.load(path))
        self.actor_old.load_state_dict(torch.load(path))

    def __update(self):
        # ACC Gradient traning
        # We have the samples why not train a bit on it?
        torch.cuda.empty_cache()

        action_log_probs, state_values, entropy = self.__eval()
        d_r = self.__discounted_rewards()
        A_T = self.__advantages(d_r, state_values)

        r_i_ts, r_i_ts_loss, a_t_hat_loss = self.__intrinsic_reward_objective()
        R_T = normalize_dist(A_T + r_i_ts)

        actor_loss = - self.__clipped_surrogate_objective(action_log_probs, R_T)  # L^CLIP

        critic_loss = (0.5 * torch.pow(state_values - d_r, 2)).mean()  # E # c1 L^VF

        entropy_bonus = entropy * self.loss_entropy_c  # c2 S[]

        curiosity_loss = (1 - (a_t_hat_loss * self.beta) + (r_i_ts_loss * self.beta))  # Li LF

        self.optimizer.zero_grad()
        # Gradient ascent -(actor_loss - critic_loss + entropy_bonus)
        total_loss = (actor_loss + critic_loss - entropy_bonus).mean() + curiosity_loss
        total_loss.backward()
        self.optimizer.step()

        print("Total ep reward: ", self.mem_buffer.rewards.sum())
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.mem_buffer.clear()

    def __eval(self):
        actions_prob = self.actor(self.mem_buffer.map_states.unsqueeze(1),
                                  self.mem_buffer.agent_states,
                                  self.env.mask)
        actions_dist = Categorical(actions_prob)
        action_log_prob = actions_dist.log_prob(self.mem_buffer.actions)
        state_values = self.critic(self.mem_buffer.map_states)
        return action_log_prob, state_values.squeeze(1), actions_dist.entropy()  # Bregman divergence

    def __advantages(self, discounted_rewards, state_values):
        advantages = torch.zeros(len(discounted_rewards))
        T = self.mem_buffer.max_length - 1
        last_state_value = state_values[T]
        t = 0
        for discounted_reward in discounted_rewards:
            advantages[t] = discounted_reward - state_values[t] + last_state_value * (
                    self.gamma ** (T - t))
            t += 1
        return advantages.float().detach()

    def __discounted_rewards(self):
        discounted_rewards = torch.zeros(len(self.mem_buffer.rewards))
        running_reward = 0
        t = 0
        for r, d in zip(reversed(self.mem_buffer.rewards), reversed(self.mem_buffer.done)):
            running_reward = r + (running_reward * self.gamma) * (1. - d)  # Zero out done states
            discounted_rewards[-t] = running_reward
            t += 1
        return discounted_rewards.float()

    def __intrinsic_reward_objective(self):
        states = self.mem_buffer.map_states.unsqueeze(1)
        next_states = self.mem_buffer.map_next_states.unsqueeze(1)
        action_probs = self.mem_buffer.action_probs
        actions = self.mem_buffer.actions

        a_t_hats, phi_t1_hats, phi_t1s, phi_ts = self.ICM(states, next_states, action_probs)
        r_i_ts = F.mse_loss(phi_t1_hats, phi_t1s, reduction='none').sum(-1)

        multi_cross_loss = 0
        for i in range(len(actions[1])):
            multi_cross_loss += F.cross_entropy(a_t_hats, actions[:, i])
        multi_cross_loss /= self.max_agents

        return (self.eta * r_i_ts).detach(), r_i_ts.mean(), multi_cross_loss / self.max_agents

    def __clipped_surrogate_objective(self, actions_log_probs, R_T):
        r_T_theta = torch.exp(actions_log_probs - self.mem_buffer.action_log_prob)
        r_T_theta = r_T_theta.mean(-1)  # average over agents
        r_T_c_theta = torch.clamp(r_T_theta, min=1 - self.eps_c, max=1 + self.eps_c)
        return torch.min(r_T_theta * R_T, r_T_c_theta * R_T).mean()  # E


# TEST
if __name__ == '__main__':
    a = torch.randn(3, 50, 50)
    a[20:50, 20:50] = 0
    b = torch.randn(3, 8, 2)
    model = ActorPolicyModel(50, 50, 29)

    log_probs = model(a, b)
    print(log_probs.shape)

    actions_dist = Categorical(log_probs)
    actions = actions_dist.sample()
    print(actions.shape)
    action_dist_log_probs = actions_dist.log_prob(actions)
    print(action_dist_log_probs)
    print(actions_dist.sample())
