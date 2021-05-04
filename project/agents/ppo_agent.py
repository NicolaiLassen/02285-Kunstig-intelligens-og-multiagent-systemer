import copy
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from agents.agent_base import BaseAgent
from environment.action import idxs_to_actions
from environment.env_wrapper import EnvWrapper
from models.curiosity import ICM
from models.policy_models import PolicyModelEncoder
from utils import mem_buffer
# PPO Actor Critic
from utils.mem_buffer import AgentMemBuffer
from utils.preprocess import normalize_dist


class PPOAgent(BaseAgent):
    mem_buffer: mem_buffer = None

    def __init__(self,
                 env: EnvWrapper,
                 actor: nn.Module,
                 critic: nn.Module,
                 optimizer: optim.Optimizer,
                 n_acc_gradient=20,
                 gamma=0.9,
                 eps_c=0.2,
                 n_max_Times_update=1):
        self.env = env
        self.actor = actor
        self.actor_old = copy.deepcopy(actor)
        self.actor_old.load_state_dict(actor.state_dict())
        self.critic = critic
        self.optimizer = optimizer
        self.action_space_n = env.action_space_n
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

    def train(self, max_Time: int, max_Time_steps: int):
        self.mem_buffer = AgentMemBuffer(max_Time)
        update_every = max_Time * self.n_max_Times_update  # TODO: BATCH
        t = 0
        s1 = self.env.reset()
        while t < max_Time_steps:
            self.save_actor()
            for ep_T in range(max_Time + 1):
                t += 1
                s = s1
                action_idxs, log_probs = self.act(s)
                actions = idxs_to_actions(action_idxs)

                print(actions[0]._name_, file=sys.stderr, flush=True)

                ## TODO
                ## FIX THIS SHIT
                temp = self.env.step(actions)
                if temp is None:
                    continue

                s1, r, d = temp

                # self.mem_buffer.set_next(s, r, actions, log_probs, d, self.mem_buffer.get_mask(d))
                # if t % update_every == 0:
                #     self.__update()

    def act(self, state):
        actions_logs_prob = self.actor_old(state[0], state[1], self.env.mask)
        actions_dist = Categorical(actions_logs_prob)
        actions = actions_dist.sample()
        action_dist_log_probs = actions_dist.log_prob(actions)
        return actions.detach(), action_dist_log_probs.detach()

    def save_actor(self):
        print("save_actor")
        # torch.save(self.actor_old.state_dict(), "encoder_actor.ckpt")

    def load_actor(self, path):
        self.actor.load_state_dict(torch.load(path))
        self.actor_old.load_state_dict(torch.load(path))

    def __update(self):
        losses_ = torch.zeros(self.n_acc_grad)  # SOME PRINT STUFF
        # ACC Gradient traning
        # We have the samples why not train a bit on it?
        for _ in range(self.n_acc_grad):
            self.optimizer.zero_grad()
            action_log_probs, state_values, entropy = self.__eval()

            A_T = self.__advantages(state_values)
            I_C_T = self.__intrinsic_curiosity(state_values)
            R_T = normalize_dist(A_T + (I_C_T * self.intrinsic_curiosity_c))

            c_s_o = self.__clipped_surrogate_objective(action_log_probs, R_T)
            loss = (-c_s_o - (entropy * self.loss_entropy_c)).mean()
            loss.backward()

            self.optimizer.step()

            # SOME PRINT STUFF
            with torch.no_grad():
                losses_[_] = loss.item()

        print("Mean ep losses: ", losses_.mean())
        print("Total ep reward: ", self.mem_buffer.rewards.sum())
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.mem_buffer.clear()

    def __eval(self):
        actions_prob = self.actor(self.mem_buffer.states, self.env.mask)
        dist = Categorical(actions_prob)
        action_log_probs = dist.log_prob(self.mem_buffer.actions)
        state_values = self.critic(self.mem_buffer.states)
        return action_log_probs, state_values, dist.entropy()  # Bregman divergence

    def __advantages(self, state_values):
        discounted_rewards = []
        running_reward = 0

        for r, d in zip(reversed(self.mem_buffer.rewards), reversed(self.mem_buffer.done)):
            running_reward = r + (running_reward * self.gamma) * (1. - d)  # Zero out done states
            discounted_rewards.append(running_reward)

        return torch.tensor(discounted_rewards, dtype=torch.float32).cuda() - state_values.detach()

    def __intrinsic_curiosity(self, state_values):
        return self.ICM(state_values)

    def __clipped_surrogate_objective(self, actions_log_probs, R_T):
        r_T_theta = torch.exp(actions_log_probs - self.mem_buffer.action_log_probs)
        r_T_c_theta = torch.clamp(r_T_theta, min=1 - self.eps_c, max=1 + self.eps_c)
        return torch.min(r_T_theta * R_T, r_T_c_theta * R_T).mean()  # E


# TEST
if __name__ == '__main__':
    a = torch.randn(10, 50, 50)
    a[20:50, 20:50] = 0
    b = torch.randn(10, 8, 2)
    model = PolicyModelEncoder(50, 50, 29)

    log_probs = model(a, b)

    actions_dist = Categorical(log_probs)
    actions = actions_dist.sample()
    action_dist_log_probs = actions_dist.log_prob(actions)

    print(actions)
