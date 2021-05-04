import torch
import torch.nn as nn
from numba import jit
from torch import Tensor

from environment.action import Action


class EnvWrapper:

    def __init__(self,
                 agents_n,
                 initial_state_m: Tensor,
                 initial_state_m_color: Tensor,
                 initial_state_v: dict,
                 goal_state_m: Tensor,
                 reward_model: nn.Module
                 ) -> None:

        self.agents = agents_n
        self.t0_map = initial_state_m
        self.t0_map_color = initial_state_m_color
        self.t0_v = initial_state_v
        self.t_T = goal_state_m
        self.reward_model = reward_model

    @jit(nopython=True)
    def step(self, actions: Action):

        if self.__is_not_applicable(actions) or \
                self.__is_conflict(actions):
            return None

        t1_map, t1_map_color, t1_state_v = self.__act(actions)

        reward = self.reward_model(t1_map)
        done = self.__check_done(t1_map)
        self.t0_map = t1_map

        return [t1_map, reward, done]

    @jit(nopython=True)
    def __check_done(self, next_state: Tensor):
        return torch.eq(next_state, self.t_T)

    @jit(nopython=True)
    def __is_not_applicable(self, actions):
        return False

    @jit(nopython=True)
    def __is_conflict(self, actions):

        for a1 in self.agents:
            for a2 in self.agents:
                print()

        return False

    @jit(nopython=True)
    def __act(self, action) -> Tensor:
        return
