import copy
import sys
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
from models.curiosity import ICM
from models.policy_models import PolicyModelEncoder
from utils import mem_buffer
# PPO Actor Critic
from utils.mem_buffer import AgentMemBuffer


# TODO: WE NEED TO BE DONE
# TODO: UPDATE WITH NEW RL IMC
# TODO: MEM BUFFER
# TODO: REWARD
# TODO: Optim code performance
# TODO: self.mem_buffer.states [map,agents], [map,colors,agents]
# TODO: CUDA

class PPOAgent(BaseAgent):
    mem_buffer: mem_buffer = None

    def __init__(self,
                 env: EnvWrapper,
                 actor: nn.Module,
                 critic: nn.Module,
                 optimizer_actor: optim.Optimizer,
                 optimizer_critic: optim.Optimizer,
                 n_acc_gradient=10,
                 gamma=0.9,
                 lamda=0.5,
                 eta=0.5,
                 beta=0.8,
                 eps_c=0.2,
                 loss_entropy_c=0.01,
                 intrinsic_curiosity_c=0.9,
                 n_max_Times_update=1):
        self.env = env
        self.actor = actor
        self.actor_old = copy.deepcopy(actor)
        self.actor_old.load_state_dict(actor.state_dict())
        self.critic = critic
        self.optimizer_actor = optimizer_actor
        self.optimizer_critic = optimizer_critic
        self.action_space_n = env.action_space_n
        # Curiosity
        self.ICM = ICM(self.action_space_n)
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
        self.intrinsic_curiosity_c = intrinsic_curiosity_c

    def train(self, max_Time: int, max_Time_steps: int):
        self.mem_buffer = AgentMemBuffer(max_Time, action_space_n=self.action_space_n)
        update_every = max_Time * self.n_max_Times_update  # TODO: BATCH
        t = 0

        while t < max_Time_steps:
            self.save_actor()
            s1 = self.env.reset()
            for ep_T in range(max_Time + 1):
                t += 1
                s = s1
                action_idxs, probs, log_prob = self.act(s)
                actions = idxs_to_actions(action_idxs)
                ## TODO
                ## FIX THIS SHIT
                print(actions[0]._name_, file=sys.stderr, flush=True)

                temp = self.env.step(actions)
                if temp is None:
                    continue

                s1, r, d = temp
                print(d, file=sys.stderr, flush=True)

                ## TODO BUFFER NEW
                # self.mem_buffer.set_next(s, s1, r, action_idxs, probs, log_prob, d, self.mem_buffer.get_mask(d))
                if t % update_every == 0:
                    self.__update()

    def act(self, state) -> Tuple[Tensor, Tensor, Tensor]:
        actions_logs_prob = self.actor_old(state[0].unsqueeze(0),
                                           state[1].unsqueeze(0),
                                           self.env.mask)
        actions_dist = Categorical(actions_logs_prob)
        actions = actions_dist.sample()
        action_dist_log_prob = actions_dist.log_prob(actions)
        return actions.detach(), actions_dist.probs.detach(), action_dist_log_prob

    def save_actor(self):
        return
        # torch.save(self.actor_old.state_dict(), "encoder_actor.ckpt")

    def load_actor(self, path):
        self.actor.load_state_dict(torch.load(path))
        self.actor_old.load_state_dict(torch.load(path))

    def __update(self):
        losses_ = torch.zeros(self.n_acc_grad)  # SOME PRINT STUFF
        # ACC Gradient traning
        # We have the samples why not train a bit on it?
        for _ in range(self.n_acc_grad):
            action_log_probs, state_values, entropy = self.__eval()

            d_r = self.__discounted_rewards()
            A_T = d_r - state_values.detach()
            r_i_ts, r_i_ts_loss, a_t_hat_loss = self.__intrinsic_reward_objective()

            R_T = A_T + (r_i_ts * self.intrinsic_curiosity_c)

            c_s_o_loss = self.__clipped_surrogate_objective(action_log_probs, R_T)

            curiosity_loss = (1 - (a_t_hat_loss * self.beta) + (r_i_ts_loss * self.beta))

            self.optimizer_actor.zero_grad()
            actor_loss = self.lamda * (- c_s_o_loss - (entropy * self.loss_entropy_c)).mean() + curiosity_loss
            actor_loss.backward()
            self.optimizer_actor.step()

            self.optimizer_critic.zero_grad()
            critic_loss = 0.5 * F.mse_loss(state_values, d_r)
            critic_loss.backward()
            self.optimizer_critic.step()

            # SOME PRINT STUFF
            with torch.no_grad():
                losses_[_] = actor_loss.item()

        print("Mean ep losses: ", losses_.mean())
        print("Total ep reward: ", self.mem_buffer.rewards.sum())
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.mem_buffer.clear()

    def __eval(self):
        ## TODO NOT python lists
        actions_prob = self.actor(self.mem_buffer.states[0],
                                  self.mem_buffer.states[1],
                                  self.env.mask)

        actions_dist = Categorical(actions_prob)
        action_log_prob = actions_dist.log_prob(self.mem_buffer.actions)
        state_values = self.critic(self.mem_buffer.states[0])
        return action_log_prob, state_values.squeeze(1), actions_dist.entropy()  # Bregman divergence

    def __discounted_rewards(self):
        discounted_rewards = []
        running_reward = 0

        for r, d in zip(reversed(self.mem_buffer.rewards), reversed(self.mem_buffer.done)):
            running_reward = r + (running_reward * self.gamma) * (1. - d)  # Zero out done states
            discounted_rewards.append(running_reward)

        return torch.stack(discounted_rewards).float().cuda()

    def __intrinsic_reward_objective(self):
        next_states = self.mem_buffer.next_states
        states = self.mem_buffer.states
        action_probs = self.mem_buffer.action_probs
        actions = self.mem_buffer.actions

        a_t_hats, phi_t1_hats, phi_t1s, phi_ts = self.ICM(states, next_states, action_probs)

        r_i_ts = F.mse_loss(phi_t1_hats, phi_t1s, reduction='none').sum(-1)

        return (self.eta * r_i_ts).detach(), r_i_ts.mean(), F.cross_entropy(a_t_hats, actions)

    def __clipped_surrogate_objective(self, actions_log_probs, R_T):
        r_T_theta = torch.exp(actions_log_probs - self.mem_buffer.action_log_prob)
        r_T_c_theta = torch.clamp(r_T_theta, min=1 - self.eps_c, max=1 + self.eps_c)
        return torch.min(r_T_theta * R_T, r_T_c_theta * R_T).mean()  # E


# TEST
if __name__ == '__main__':
    a = torch.randn(3, 50, 50)
    a[20:50, 20:50] = 0
    b = torch.randn(3, 8, 2)
    model = PolicyModelEncoder(50, 50, 29)

    log_probs = model(a, b)
    print(log_probs.shape)

    actions_dist = Categorical(log_probs)
    actions = actions_dist.sample()
    print(actions.shape)
    action_dist_log_probs = actions_dist.log_prob(actions)
    print(action_dist_log_probs)
