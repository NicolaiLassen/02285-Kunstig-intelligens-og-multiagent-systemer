from typing import List

import torch
from numba import jit
from torch import Tensor

from environment.action import Action, ActionType
from utils.preprocess import Entity, LevelState


class EnvWrapper:

    # agents = [Entity]
    # matrix = [[Entity]]

    def __init__(self,
                 initial_state: LevelState,
                 goal_state: LevelState,
                 agents_n: int,
                 action_space_n: int,
                 initial_state_m: Tensor,
                 initial_state_m_color: Tensor,
                 initial_agent_places: List[Entity],
                 initial_box_places: List[Entity],
                 goal_state_m: Tensor,
                 reward_func,
                 mask=None) -> None:

        self.initial_state = initial_state
        self.t0_state = initial_state

        self.initial_state_m = initial_state_m
        self.initial_state_m_color = initial_state_m_color
        self.initial_agent_places = initial_agent_places
        self.initial_box_places = initial_box_places

        self.agents_n = agents_n
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

        for action in actions:
            if not self.__is_applicable(action):
                return None

        if self.__is_conflict(actions):
            return None

        t1_map, t1_map_color = self.__act(actions)

        reward = self.reward_func(t1_map)
        done = self.__check_done(t1_map)
        self.t0_map = t1_map

        return t1_map, reward, done

    def reset(self):
        self.t0_map = self.initial_state_m
        self.t0_map_color = self.initial_state_m_color
        self.t0_agent_places = self.initial_agent_places
        self.t0_box_places = self.initial_box_places
        return

    @jit(nopython=True)
    def __check_done(self, next_state: Tensor):
        return torch.eq(next_state, self.t_T)

    @jit(nopython=True)
    def __is_applicable(self, index: int, action: Action):
        agent_row, agent_col = self.__agent_row_col(index)

        if action.type is ActionType.NoOp:
            return True
        elif action.type is ActionType.Move:
            # check that next position is free
            next_agent_row = agent_row + action.agent_row_delta
            next_agent_col = agent_col + action.agent_col_delta
            return self.__is_free(next_agent_row, next_agent_col)
        elif action.type is ActionType.Push:
            # check that next agent position is box
            next_agent_row = agent_row + action.agent_row_delta
            next_agent_col = agent_col + action.agent_col_delta
            if not self.__is_box(next_agent_row, next_agent_col):
                return False
            # check that agent and box is same color
            if not self.__is_same_color(agent_row, agent_col, next_agent_row, next_agent_col):
                return False
            # check that next box position is free
            next_box_row = next_agent_row + action.box_row_delta
            next_box_col = next_agent_col + action.box_col_delta
            return self.__is_free(next_box_row, next_box_col)

        elif action.type is ActionType.Pull:
            # check that next agent position is free
            next_agent_row = agent_row + action.agent_row_delta
            next_agent_col = agent_col + action.agent_col_delta
            if not self.__is_free(next_agent_row, next_agent_col):
                return False

            # check that next box position is free
            box_row = next_agent_row + (action.box_row_delta * -1)
            box_col = next_agent_col + (action.box_col_delta * -1)
            if not self.__is_box(box_row, box_col):
                return False

            # check that agent and box is same color
            return self.__is_same_color(agent_row, agent_col, box_row, box_col)
        else:
            return False

    @jit(nopython=True)
    def __is_conflict(self, actions: List[Action]):
        num_agents = self.agents_n

        next_agent_rows = [0 for _ in range(num_agents)]
        next_agent_cols = [0 for _ in range(num_agents)]
        box_rows = [0 for _ in range(num_agents)]
        box_cols = [0 for _ in range(num_agents)]

        for i in range(num_agents):
            agent_row, agent_col = self.__agent_row_col(i)
            action = actions[i]
            if action.type is ActionType.NoOp:
                continue
            elif action.type is ActionType.Move:
                next_agent_rows[i] = agent_row + action.agent_row_delta
                next_agent_cols[i] = agent_col + action.agent_col_delta
            elif action.type is ActionType.Push:
                next_agent_rows[i] = agent_row + action.agent_row_delta
                next_agent_cols[i] = agent_col + action.agent_col_delta
                box_rows[i] = next_agent_rows[i]
                box_cols[i] = next_agent_cols[i]
            elif action.type is ActionType.Pull:
                next_agent_rows[i] = agent_row + action.agent_row_delta
                next_agent_cols[i] = agent_col + action.agent_col_delta
                box_rows[i] = agent_row + (action.box_row_delta * -1)
                box_cols[i] = agent_col + (action.box_col_delta * -1)

        for a1 in range(num_agents):
            if actions[a1].type is ActionType.NoOp:
                continue

            for a2 in range(num_agents):
                if actions[a2].type is ActionType.NoOp:
                    continue

                # is moving same box
                if box_rows[a1] == box_rows[a2] and box_cols[a1] == box_cols[a2]:
                    return True

                # is moving into same position
                if next_agent_rows[a1] == next_agent_rows[a2] and next_agent_cols[a1] == next_agent_cols[a2]:
                    return True

        return False

    def __is_same_color(self, a_row, a_col, b_row, b_col):
        return self.t0_state.matrix[a_row][a_col].color == self.t0_state.matrix[b_row][b_col].color

    def __is_box(self, row, col):
        return self.t0_state.matrix[row][col].is_box()

    def __is_free(self, row, col):
        return self.t0_state.matrix[row][col].is_free()

    def __agent_row_col(self, index: int):
        agent = self.t0_state.agents[index]
        return agent.row, agent.col

    @jit(nopython=True)
    def __act(self, action) -> Tensor:
        return
