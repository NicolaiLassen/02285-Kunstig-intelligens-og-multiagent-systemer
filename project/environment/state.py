import sys
from copy import deepcopy
from typing import List

import numpy as np

from environment.action import ActionType, Action


class Constraint:
    def __init__(self, agent, state, t, agents):
        self.agent: int = agent
        self.agents: [int] = agents
        self.state: State = state
        self.t = t


class State:
    free_value = 32  # ord(" ")
    agent_0_value = 48  # ord("0")
    agent_9_value = 57  # ord("9")
    box_a_value = 65  # ord("A")
    box_z_value = 90  # ord("Z")

    agent = 0
    g = 0
    h = 0
    parent = None
    action = None

    def __init__(self, map, agent, colors, agents, goal_state_positions):
        self.map: np.ndarray = map
        self.agent = agent
        self.colors: np.ndarray = colors
        self.agents: np.ndarray = agents
        self.goal_state_positions = goal_state_positions

    def __act(self, action: Action, index: int) -> 'State':

        next_state = deepcopy(self)

        # Update agent location
        prev_agent_row, prev_agent_col = self.agent_row_col(index)
        next_agent_row = prev_agent_row + action.agent_row_delta
        next_agent_col = prev_agent_col + action.agent_col_delta
        agent_value = self.map[prev_agent_row][prev_agent_col]
        agent_color = self.colors[prev_agent_row][prev_agent_col]
        next_state.agents[index] = np.asarray([next_agent_row, next_agent_col])

        # Update level matrices and agent pos
        if action.type is ActionType.NoOp:
            return next_state
        elif action.type is ActionType.Move:
            next_state.map[next_agent_row][next_agent_col] = agent_value
            next_state.map[prev_agent_row][prev_agent_col] = self.free_value

            next_state.colors[next_agent_row][next_agent_col] = agent_color
            next_state.colors[prev_agent_row][prev_agent_col] = 0

        elif action.type is ActionType.Push:
            box_value = self.map[next_agent_row][next_agent_col]
            box_color = self.colors[next_agent_row][next_agent_col]
            next_box_row = next_agent_row + action.box_row_delta
            next_box_col = next_agent_col + action.box_col_delta

            next_state.map[next_box_row][next_box_col] = box_value
            next_state.map[next_agent_row][next_agent_col] = agent_value
            next_state.map[prev_agent_row][prev_agent_col] = self.free_value

            next_state.colors[next_box_row][next_box_col] = box_color
            next_state.colors[next_agent_row][next_agent_col] = agent_color
            next_state.colors[prev_agent_row][prev_agent_col] = 0

        elif action.type is ActionType.Pull:
            prev_box_row = prev_agent_row + (action.box_row_delta * -1)
            prev_box_col = prev_agent_col + (action.box_col_delta * -1)
            box_value = self.map[prev_box_row][prev_box_col]
            box_color = self.colors[prev_box_row][prev_box_col]

            next_state.map[next_agent_row][next_agent_col] = agent_value
            next_state.map[prev_agent_row][prev_agent_col] = box_value
            next_state.map[prev_box_row][prev_box_col] = self.free_value

            next_state.colors[next_agent_row][next_agent_col] = agent_color
            next_state.colors[prev_agent_row][prev_agent_col] = box_color
            next_state.colors[prev_box_row][prev_box_col] = 0

        next_state.parent = self
        next_state.action = action
        next_state.g = self.g + 1
        next_state.h = len(self.goal_state_positions) - self.__count_goals()

        return next_state

    def get_expanded_states(self, constraints: List[Constraint]) -> '[State, ...]':
        index = self.agent
        expanded_states = []
        applicable_actions = [action for action in Action if self.__is_applicable(index, action)]

        for action in applicable_actions:
            expanded_states.append(self.__act(action, index))

        for contraint in constraints:
            if contraint.agent != index:
                continue
            if contraint.t != self.g:
                continue
            for state in expanded_states:
                if state == contraint.state:
                    expanded_states.remove(state)

        return expanded_states

    def extract_plan(self) -> '[State, ...]':
        plan = [None for _ in range(self.g)]
        state = self
        while state.action is not None:
            plan[state.g - 1] = state
            state = state.parent
        return plan

    def is_goal_state(self) -> 'bool':
        return len(self.goal_state_positions) == self.__count_goals()

    def agent_row_col(self, index: int):
        agent_position = self.agents[index]
        return int(agent_position[0]), int(agent_position[1])

    def __is_applicable(self, index: int, action: Action) -> bool:
        agent_row, agent_col = self.agent_row_col(index)

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
            # check that box position is box
            box_row = agent_row + (action.box_row_delta * -1)
            box_col = agent_col + (action.box_col_delta * -1)
            if not self.__is_box(box_row, box_col):
                return False
            # check that agent and box is same color
            return self.__is_same_color(agent_row, agent_col, box_row, box_col)
        else:
            return False

    def __count_goals(self):
        goal_count = 0
        for row in range(len(self.map)):
            for col in range(len(self.map[row])):
                key = str([row, col])
                if key not in self.goal_state_positions:
                    continue
                val = self.map[row][col].item()
                goal_count += 1 if self.goal_state_positions[key] == val else 0
        return goal_count

    def __is_same_color(self, a_row, a_col, b_row, b_col) -> bool:
        return self.colors[a_row][a_col] == self.colors[b_row][b_col]

    def __is_box(self, row, col) -> bool:
        return self.box_a_value <= self.map[row][col] <= self.box_z_value

    def __is_free(self, row, col) -> bool:
        return self.map[row][col].item() == self.free_value

    def f(self):
        return self.g + self.h

    def __hash__(self):
        prime = 31
        _hash = 1
        _hash = _hash * prime + hash(str((self.map)))
        return _hash

    def __eq__(self, other: 'State'):
        return self.__hash__() == other.__hash__()

    def __lt__(self, other: 'State'):
        return self.f() < other.f()
