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
        self.free_value = 32  # ord(" ")
        self.agent_0_value = 48  # ord("0")
        self.agent_9_value = 57  # ord("9")
        self.box_a_value = 65  # ord("A")
        self.box_z_value = 90  # ord("Z")

        self.action_space_n = action_space_n
        self.initial_state = initial_state
        self.goal_state = goal_state

        self.agents_n = initial_state.agents_n
        self.rows_n = initial_state.rows_n
        self.cols_n = initial_state.cols_n

        self.goal_state_positions = {}
        for row in range(len(self.goal_state.level)):
            for col in range(len(self.goal_state.level[row])):
                val = goal_state.level[row][col]
                if self.box_a_value <= val <= self.box_z_value:
                    self.goal_state_positions[val.item()] = [row, col]
                if self.agent_0_value <= val <= self.agent_9_value:
                    self.goal_state_positions[val.item()] = [row, col]

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
                # print('# action not applicable\n{}: {}'.format(index, action), file=sys.stderr, flush=True)
                return None

        if self.__is_conflict(actions):
            # print('# actions contain conflict\n{}'.format(actions), file=sys.stderr, flush=True)
            return None

        t1_state = self.__act(actions)
        done = self.__check_done(t1_state)
        reward = self.reward(t1_state)
        return [t1_state.level, t1_state.agents], reward, done

    def reward(self, state) -> int:

        sum_distance = 0
        for row in range(len(state.level)):
            for col in range(len(state.level[row])):
                val = state.level[row][col].item()
                if val not in self.goal_state_positions:
                    continue
                goal_row, goal_col = self.goal_state_positions[val]
                distance = abs(goal_row - row) + abs(goal_col - col)
                sum_distance += distance

        return sum_distance * -1

    # def reward(self, state) -> int:
    #     sum_agent_distance = 0
    #     for i in range(self.agents_n):
    #         a_row, a_col = state.agent_row_col(i)
    #         b_row, b_col = self.goal_state.agent_row_col(i)
    #         agent_distance = abs(b_row - a_row) + abs(b_col - a_col)
    #         sum_agent_distance += agent_distance
    #
    #     return sum_agent_distance * -1

    def reset(self) -> List[Tensor]:
        self.t0_state = self.initial_state
        return [self.t0_state.level, self.t0_state.agents]

    def __check_done(self, state: LevelState) -> bool:
        return torch.equal(state.level, self.goal_state.level)

    def __is_applicable(self, index: int, action: Action):
        agent_row, agent_col = self.t0_state.agent_row_col(index)

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
                # print('# next agent position is NOT free', file=sys.stderr, flush=True)
                return False
            # check that box position is box
            box_row = agent_row + (action.box_row_delta * -1)
            box_col = agent_col + (action.box_col_delta * -1)
            if not self.__is_box(box_row, box_col):
                # print('# box position is NOT box', file=sys.stderr, flush=True)
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
            agent_row, agent_col = self.t0_state.agent_row_col(i)
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
                    # print('# actions.conflict\nis moving same box', file=sys.stderr, flush=True)
                    return True

                # is moving into same position
                if next_agent_rows[a1] == next_agent_rows[a2] and next_agent_cols[a1] == next_agent_cols[a2]:
                    # print('# actions.conflict\nis moving into same position', file=sys.stderr, flush=True)
                    return True

        return False

    def __is_same_color(self, a_row, a_col, b_row, b_col):
        return self.t0_state.colors[a_row][a_col] == self.t0_state.colors[b_row][b_col]

    def __is_box(self, row, col):
        return self.box_a_value <= self.t0_state.level[row][col] <= self.box_z_value

    def __is_free(self, row, col):
        return self.t0_state.level[row][col].item() == self.free_value

    def __act(self, actions: List[Action]) -> LevelState:

        next_state = self.t0_state

        for index, action in enumerate(actions):
            # Update agent location
            prev_agent_row, prev_agent_col = self.t0_state.agent_row_col(index)
            next_agent_row = prev_agent_row + action.agent_row_delta
            next_agent_col = prev_agent_col + action.agent_col_delta
            agent_value = self.t0_state.level[prev_agent_row][prev_agent_col]
            next_state.agents[index] = torch.tensor([next_agent_row, next_agent_col])

            # Update level matrix
            if action.type is ActionType.NoOp:
                continue
            elif action.type is ActionType.Move:
                next_state.level[next_agent_row][next_agent_col] = agent_value
                next_state.level[prev_agent_row][prev_agent_col] = self.free_value
            elif action.type is ActionType.Push:
                box_value = self.t0_state.level[next_agent_row][next_agent_col]
                next_box_row = next_agent_row + action.box_row_delta
                next_box_col = next_agent_col + action.box_col_delta
                next_state.level[next_box_row][next_box_col] = box_value
                next_state.level[next_agent_row][next_agent_col] = agent_value
                next_state.level[prev_agent_row][prev_agent_col] = self.free_value
            elif action.type is ActionType.Pull:
                prev_box_row = prev_agent_row + (action.box_row_delta * -1)
                prev_box_col = prev_agent_col + (action.box_col_delta * -1)
                box_value = self.t0_state.level[prev_box_row][prev_box_col]
                next_state.level[next_agent_row][next_agent_col] = agent_value
                next_state.level[prev_agent_row][prev_agent_col] = box_value
                next_state.level[prev_box_row][prev_box_col] = self.free_value
        return next_state
