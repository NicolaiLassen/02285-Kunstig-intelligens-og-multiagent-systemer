import sys
from typing import List

import torch
from numba import jit
from torch import Tensor

from environment.action import Action


def debug_print(message):
    print(message, file=sys.stderr, flush=True)


from utils.preprocess import Entity


class EnvWrapper:

    def __init__(self,
                 agents_n: int,
                 action_space_n: int,
                 initial_state_m: Tensor,
                 initial_state_m_color: Tensor,
                 initial_agent_places: List[Entity],
                 initial_box_places: List[Entity],
                 goal_state_m: Tensor,
                 reward_func,
                 mask=None
                 ) -> None:

        self.agents = agents_n
        self.action_space_n = action_space_n
        self.t0_map = initial_state_m
        self.t0_map_color = initial_state_m_color

        self.t0_agent_places = initial_agent_places
        self.t0_box_places = initial_box_places

        self.t_T = goal_state_m
        self.reward_func = reward_func
        self.mask = mask

    @jit(nopython=True)
    def step(self, actions: List[Action]):

        if self.__is_not_applicable(actions) or \
                self.__is_conflict(actions):
            return None

        t1_map, t1_map_color = self.__act(actions)

        reward = self.reward_func(t1_map)
        done = self.__check_done(t1_map)
        self.t0_map = t1_map

        return [t1_map, reward, done]

    def reset(self):
        return

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
