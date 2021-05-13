import copy
import os
import pickle
from random import randint
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.distributions import Categorical

from environment.action import idxs_to_actions
from environment.env_wrapper import EnvWrapper
from utils import mem_buffer
# PPO Actor Critic
from utils.mem_buffer import AgentMemBuffer
from utils.normalize_dist import normalize_dist


def delete_file(path):
    if os.path.exists(path):
        os.remove(path)


levels_n = 21


def random_level():
    return randint(0, levels_n)


class PPOAgent():
    mem_buffer: mem_buffer = None

    # counters ckpt
    t_update = 0  # t * 1000
    model_save_every = 50  # (8000000/4)  / 2000 / 50

    intrinsic_reward_ckpt = []
    curiosity_loss_ckpt = []
    actor_loss_ckpt = []
    critic_loss_ckpt = []
    reward_level_ckpt = {i: [] for i in range(levels_n)}

    def __init__(self,
                 env: EnvWrapper,
                 actor: nn.Module,
                 critic: nn.Module,
                 curiosity: nn.Module = None,
                 optimizer: optim.Optimizer = None,
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
        self.curiosity = curiosity
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
        level = random_level()
        level_running_reward = 0
        self.env.load(level)
        print(self.env)
        t = 0
        ep_t = 0
        s1 = self.env.reset()
        while t < max_Time_steps:
            while ep_t < max_Time:
                s = s1
                action_idxs, probs, log_prob = self.act(s[0].cuda(),
                                                        self.env.goal_state.level.float().cuda(),
                                                        s[1].cuda(),
                                                        s[2].cuda())
                actions = idxs_to_actions(action_idxs)
                temp_step = self.env.step(actions)
                if temp_step is None:
                    continue

                t += 1
                ep_t += 1
                s1, r, d = temp_step
                level_running_reward += r
                self.mem_buffer.set_next(s, s1, self.env.goal_state.level.float(), r, action_idxs, probs, log_prob, d)
                if d:
                    self.reward_level_ckpt[level].append(level_running_reward)
                    level = random_level()
                    self.env.load(level)
                    s1 = self.env.reset()
                    level_running_reward = 0

            level = random_level()
            self.env.load(level)
            s1 = self.env.reset()
            self.__update()
            ep_t = 0
            level_running_reward = 0

    def act(self, map_state: Tensor, map_goal_state: Tensor, color_state: Tensor, agent_state: Tensor) -> Tuple[
        Tensor, Tensor, Tensor]:
        map_state = normalize_dist(map_state)
        agent_state = normalize_dist(agent_state)
        color_state = normalize_dist(color_state)
        actions_logs_prob = self.actor_old(map_state.unsqueeze(0).unsqueeze(0),
                                           map_goal_state.unsqueeze(0).unsqueeze(0),
                                           color_state.unsqueeze(0).unsqueeze(0),
                                           agent_state.unsqueeze(0))

        actions_dist = Categorical(actions_logs_prob)
        actions = actions_dist.sample()
        action_dist_log_prob = actions_dist.log_prob(actions)
        return actions.detach(), actions_dist.probs.detach(), action_dist_log_prob.detach()

    def save_ckpt(self):
        if self.t_update % self.model_save_every == 0:
            torch.save(self.actor_old.state_dict(), "ckpt/actor_{}.ckpt".format(self.t_update))
        torch.save(torch.tensor(self.curiosity_loss_ckpt), "ckpt/losses_curiosity.ckpt")
        torch.save(torch.tensor(self.intrinsic_reward_ckpt), "ckpt/intrinsic_rewards.ckpt")
        torch.save(torch.tensor(self.actor_loss_ckpt), "ckpt/losses_actor.ckpt")
        torch.save(torch.tensor(self.critic_loss_ckpt), "ckpt/losses_critic.ckpt")

        delete_file('ckpt/reward_level.ckpt')
        with open('ckpt/reward_level.ckpt', 'wb') as handle:
            pickle.dump(self.reward_level_ckpt, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.t_update += 1

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

        with torch.no_grad():
            self.intrinsic_reward_ckpt.append(r_i_ts.sum().item())
            self.curiosity_loss_ckpt.append(curiosity_loss.item())
            self.actor_loss_ckpt.append(actor_loss.sum().item())
            self.critic_loss_ckpt.append(critic_loss.sum().item())
            self.save_ckpt()

        self.actor_old.load_state_dict(self.actor.state_dict())
        self.mem_buffer.clear()

    def __eval(self):
        actions_prob = self.actor(self.mem_buffer.map_states.unsqueeze(1),
                                  self.mem_buffer.map_goal_states.unsqueeze(1),
                                  self.mem_buffer.map_color_states.unsqueeze(1),
                                  self.mem_buffer.agent_states)
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
        return advantages.float().cuda().detach()

    def __discounted_rewards(self):
        discounted_rewards = torch.zeros(len(self.mem_buffer.rewards))
        running_reward = 0
        t = 0
        for r, d in zip(reversed(self.mem_buffer.rewards), reversed(self.mem_buffer.done)):
            running_reward = r + (
                    running_reward * self.gamma)  # We wan't done rewards due to sparse env # * (1. - d)
            discounted_rewards[-t] = running_reward
            t += 1
        return discounted_rewards.float().cuda()

    def __intrinsic_reward_objective(self):
        states = self.mem_buffer.map_states.unsqueeze(1)
        next_states = self.mem_buffer.map_next_states.unsqueeze(1)
        action_probs = self.mem_buffer.action_probs
        actions = self.mem_buffer.actions

        a_t_hats, phi_t1_hats, phi_t1s, phi_ts = self.curiosity(states, next_states, action_probs)
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
