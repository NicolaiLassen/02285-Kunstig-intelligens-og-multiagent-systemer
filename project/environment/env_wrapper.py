import copy
from typing import List, Tuple, Optional

import torch
from torch import Tensor

from environment.action import Action, ActionType
from environment.level_loader import load_level
from utils.preprocess import LevelState


def debug_print(s):
    # print(s, file=sys.stderr, flush=True)
    return


class EnvWrapper:
    initial_state = None
    goal_state = None
    t0_state = None
    agents_n = 0
    rows_n = 0
    cols_n = 0
    step_n = 0
    goal_state_positions = {}

    def __init__(
            self,
            action_space_n: int = 29,
    ) -> None:
        self.free_value = 32  # ord(" ")
        self.agent_0_value = 48  # ord("0")
        self.agent_9_value = 57  # ord("9")
        self.box_a_value = 65  # ord("A")
        self.box_z_value = 90  # ord("Z")

        self.action_space_n = action_space_n

    def __repr__(self):
        return self.t0_state.__repr__()

    def load(self, i: int):
        initial_state, goal_state = load_level(i)
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

        self.t0_state = copy.deepcopy(initial_state)
        self.step_n = 0

    def step(self, actions: List[Action]) -> Optional[Tuple[List[Tensor], int, bool]]:

        for index, action in enumerate(actions):
            if not self.__is_applicable(index, action):
                debug_print('# action not applicable\n{}: {}'.format(index, action))
                return None

        if self.__is_conflict(actions):
            debug_print('# actions contain conflict\n{}'.format(actions))
            return None

        t1_state = self.__act(actions)
        done = self.__check_done(t1_state)
        reward = self.reward(t1_state)
        self.t0_state = t1_state
        self.step_n += 1

        if done:
            self.reset()

        return [t1_state.level.float(), t1_state.agents.float()], reward, done

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

        return (sum_distance * -1) - self.step_n

    def reset(self) -> List[Tensor]:
        self.step_n = 0
        self.t0_state = copy.deepcopy(self.initial_state)
        return [self.t0_state.level.float(), self.t0_state.agents.float()]

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
                debug_print('# next agent position is NOT box')
                return False
            # check that agent and box is same color
            if not self.__is_same_color(agent_row, agent_col, next_agent_row, next_agent_col):
                debug_print('# box and agent is NOT same color')
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
                debug_print('# next agent position is NOT free')
                return False
            # check that box position is box
            box_row = agent_row + (action.box_row_delta * -1)
            box_col = agent_col + (action.box_col_delta * -1)
            if not self.__is_box(box_row, box_col):
                debug_print('# box position is NOT box')
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
                    debug_print('# actions.conflict\nis moving same box')
                    return True

                # is moving into same position
                if next_agent_rows[a1] == next_agent_rows[a2] and next_agent_cols[a1] == next_agent_cols[a2]:
                    debug_print('# actions.conflict\nis moving into same position')
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
            agent_color = self.t0_state.colors[prev_agent_row][prev_agent_col]
            next_state.agents[index] = torch.tensor([next_agent_row, next_agent_col])

            # Update level matrix
            if action.type is ActionType.NoOp:
                continue
            elif action.type is ActionType.Move:
                next_state.level[next_agent_row][next_agent_col] = agent_value
                next_state.level[prev_agent_row][prev_agent_col] = self.free_value

                next_state.colors[next_agent_row][next_agent_col] = agent_color
                next_state.colors[prev_agent_row][prev_agent_col] = 0

            elif action.type is ActionType.Push:
                box_value = self.t0_state.level[next_agent_row][next_agent_col]
                box_color = self.t0_state.colors[next_agent_row][next_agent_col]
                next_box_row = next_agent_row + action.box_row_delta
                next_box_col = next_agent_col + action.box_col_delta

                next_state.level[next_box_row][next_box_col] = box_value
                next_state.level[next_agent_row][next_agent_col] = agent_value
                next_state.level[prev_agent_row][prev_agent_col] = self.free_value

                next_state.colors[next_box_row][next_box_col] = box_color
                next_state.colors[next_agent_row][next_agent_col] = agent_color
                next_state.colors[prev_agent_row][prev_agent_col] = 0

            elif action.type is ActionType.Pull:
                prev_box_row = prev_agent_row + (action.box_row_delta * -1)
                prev_box_col = prev_agent_col + (action.box_col_delta * -1)
                box_value = self.t0_state.level[prev_box_row][prev_box_col]
                box_color = self.t0_state.colors[prev_box_row][prev_box_col]

                next_state.level[next_agent_row][next_agent_col] = agent_value
                next_state.level[prev_agent_row][prev_agent_col] = box_value
                next_state.level[prev_box_row][prev_box_col] = self.free_value

                next_state.colors[next_agent_row][next_agent_col] = agent_color
                next_state.colors[prev_agent_row][prev_agent_col] = box_color
                next_state.colors[prev_box_row][prev_box_col] = 0

        return next_state
