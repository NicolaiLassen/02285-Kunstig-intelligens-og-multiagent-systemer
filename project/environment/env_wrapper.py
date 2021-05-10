import sys
from typing import List, Tuple, Optional

import torch
from torch import Tensor

from environment.action import Action, ActionType
from utils.preprocess import LevelState


# TODO convert list to tensor with ORD

class EnvWrapper:

    def __init__(
            self,
            action_space_n: int,
            initial_state: LevelState,
            goal_state: LevelState
    ) -> None:
        self.free_value = float(32.)  # ord(" ")
        self.box_a_value = float(65.)  # ord("A")
        self.box_z_value = float(90.)  # ord("Z")

        self.action_space_n = action_space_n
        self.initial_state = initial_state
        self.goal_state = goal_state

        self.agents_n = initial_state.agents_n
        self.rows_n = initial_state.rows_n
        self.cols_n = initial_state.cols_n

        self.mask = None

        #        self.initial_state = initial_state
        #        self.goal_state = goal_state

        self.t0_state = initial_state

    def __repr__(self):
        return self.t0_state.__repr__()

    def step(self, actions: List[Action]) -> Optional[Tuple[List[Tensor], int, bool]]:

        # TODO FIX NONE
        for index, action in enumerate(actions):
            if not self.__is_applicable(index, action):
                # print('#action not applicable\n{}: {}'.format(index, action), file=sys.stderr, flush=True)
                return None

        if self.__is_conflict(actions):
            # print('#actions contain conflict\n{}'.format(actions), file=sys.stderr, flush=True)
            return None

        level_s1, agents_s1 = self.__act(actions)
        done = self.__check_done(level_s1)
        reward = 0
        return [level_s1, agents_s1], reward, done

    def reset(self) -> List[Tensor]:
        self.t0_state = self.initial_state
        return [self.t0_state.level, self.t0_state.agents]

    def __check_done(self, s1: Tensor) -> bool:
        return torch.equal(s1, self.goal_state.level)

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

    def __is_conflict(self, actions: List[Action]):
        num_agents = self.agents_n

        next_agent_rows = [-1 for _ in range(num_agents)]
        next_agent_cols = [-1 for _ in range(num_agents)]
        box_rows = [-1 for _ in range(num_agents)]
        box_cols = [-1 for _ in range(num_agents)]

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
                if a1 == a2:
                    continue
                if actions[a2].type is ActionType.NoOp:
                    continue

                # is moving same box
                if box_rows[a1] == box_rows[a2] and box_cols[a1] == box_cols[a2]:
                    print('# actions.conflict\nis moving same box', file=sys.stderr, flush=True)
                    return True

                # is moving into same position
                if next_agent_rows[a1] == next_agent_rows[a2] and next_agent_cols[a1] == next_agent_cols[a2]:
                    print('# actions.conflict\nis moving into same position', file=sys.stderr, flush=True)
                    return True

        return False

    def __is_same_color(self, a_row, a_col, b_row, b_col):
        # TODO add color tensor
        return True

    def __is_box(self, row, col):
        return self.box_a_value <= self.t0_state.level[row][col] <= self.box_z_value

    def __is_free(self, row, col):
        return self.t0_state.level[row][col] == self.free_value

    def __agent_row_col(self, index: int):
        agent_position = self.t0_state.agents[index]
        return agent_position[0], agent_position[1]

    def __act(self, actions: List[Action]) -> Tuple[Tensor, Tensor]:
        for index, action in enumerate(actions):
            # Update agent location
            agent_positions = self.t0_state.agents[index]
            prev_agent_row = agent_positions[0]
            prev_agent_col = agent_positions[1]
            next_agent_row = prev_agent_row + action.agent_row_delta
            next_agent_col = prev_agent_col + action.agent_col_delta
            agent_positions[0] = next_agent_row
            agent_positions[1] = next_agent_col
            self.t0_state.agents[index] = agent_positions

            agent_value = self.t0_state.level[prev_agent_row][prev_agent_col]

            # Update level matrix
            if action.type is ActionType.NoOp:
                continue
            elif action.type is ActionType.Move:
                self.t0_state.level[next_agent_row][next_agent_col] = agent_value
                self.t0_state.level[prev_agent_row][prev_agent_col] = self.free_value
            elif action.type is ActionType.Push:
                box_value = self.t0_state.level[next_agent_row][next_agent_col]
                next_box_row = next_agent_row + action.box_row_delta
                next_box_col = next_agent_col + action.box_col_delta
                self.t0_state.level[next_box_row][next_box_col] = box_value
                self.t0_state.level[next_agent_row][next_agent_col] = agent_value
                self.t0_state.level[prev_agent_row][prev_agent_col] = self.free_value
            elif action.type is ActionType.Pull:
                prev_box_row = prev_agent_row + (action.box_row_delta * -1)
                prev_box_col = prev_agent_col + (action.box_col_delta * -1)
                box_value = self.t0_state.level[prev_box_row][prev_box_col]
                self.t0_state.level[prev_agent_row][prev_agent_col] = box_value
                self.t0_state.level[next_agent_row][next_agent_col] = agent_value
                self.t0_state.level[prev_box_row][prev_box_col] = self.free_value
        return self.t0_state.level, self.t0_state.agents
