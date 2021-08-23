import sys
from copy import deepcopy
from typing import List

from src.action import ActionType, Action
from src.position import Position


class Constraint:
    def __init__(self, agent, state, other_state, t):
        self.agent: int = agent
        self.state: State = state
        self.other_state: State = other_state
        self.t = t


class Conflict:
    def __init__(self, agents, states, position, step):
        self.agents: [int] = agents
        self.states = states
        self.position = position
        self.step = step

    def __repr__(self):
        return 'Conflict!\nAgent: {} v {}\nPosition: {},{} step: {}\n' \
            .format(self.agents[0], self.agents[1], self.position[0], self.position[1], self.step)


class State:
    _hash: int = None

    map: List[List[str]]
    agent: str
    agent_pos: Position
    goal_state_positions: dict

    action: Action = None
    parent: 'State'
    g: int
    h: int
    f: int

    def __init__(self, map, agent, agent_row, agent_col):
        self.map = map
        self.agent = agent
        self.agent_row = agent_row
        self.agent_col = agent_col
        self.g = 0
        self.h = 0
        self.f = self.g + self.h

    def box_row(self):
        # if self.action.type is ActionType.Push:
        #     return self.agent_row + self.action.box_row_delta
        # if self.action.type is ActionType.Pull:
        #     return self.agent_row - self.action.box_row_delta
        return -1

    def box_col(self):
        # if self.action.type is ActionType.Push:
        #     return self.agent_col + self.action.box_col_delta
        # if self.action.type is ActionType.Pull:
        #     return self.agent_col - self.action.box_col_delta
        return -1

    def get_solution(self) -> '[AState, ...]':
        plan = [None for _ in range(self.g)]
        state = self
        while state.action is not None:
            plan[state.g - 1] = state
            state = state.parent
        return plan

    def expand_state(self, constraints: List[Constraint]):

        ## TODO THIS WORKS BUT MAKE IT BETTER!
        ## DOES NOT WORK IN ALL CASES PLZ

        applicable_actions = [action for action in Action if self.is_applicable(action)]
        expanded_states = [self.act(action) for action in applicable_actions]

        remove_index = []
        for i, s in enumerate(expanded_states):
            for constraint in constraints:
                if constraint.t != self.g:
                    continue
                if s == constraint.state:
                    remove_index.append(i)
        filtered_states = [s for i, s in enumerate(expanded_states) if i not in remove_index]

        return filtered_states

    def is_goal_state(self) -> bool:
        return len(self.goal_state_positions) == self.__count_goals()

    def act(self, action: Action) -> 'State':
        next_state = deepcopy(self)

        # Update agent location
        prev_agent_row = self.agent_row
        prev_agent_col = self.agent_col
        next_agent_row = prev_agent_row + action.agent_row_delta
        next_agent_col = prev_agent_col + action.agent_col_delta
        agent_value = self.map[prev_agent_row][prev_agent_col]

        next_state.agent_row = next_agent_row
        next_state.agent_col = next_agent_col

        # Update level matrices and agent pos
        if action.type is ActionType.NoOp:
            return next_state
        elif action.type is ActionType.Move:
            next_state.map[next_agent_row][next_agent_col] = agent_value
            next_state.map[prev_agent_row][prev_agent_col] = " "

        elif action.type is ActionType.Push:
            box_value = self.map[next_agent_row][next_agent_col]
            next_box_row = next_agent_row + action.box_row_delta
            next_box_col = next_agent_col + action.box_col_delta

            next_state.map[next_box_row][next_box_col] = box_value
            next_state.map[next_agent_row][next_agent_col] = agent_value
            next_state.map[prev_agent_row][prev_agent_col] = " "

        elif action.type is ActionType.Pull:
            prev_box_row = prev_agent_row + (action.box_row_delta * -1)
            prev_box_col = prev_agent_col + (action.box_col_delta * -1)
            box_value = self.map[prev_box_row][prev_box_col]

            next_state.map[next_agent_row][next_agent_col] = agent_value
            next_state.map[prev_agent_row][prev_agent_col] = box_value
            next_state.map[prev_box_row][prev_box_col] = " "

        next_state.parent = self
        next_state.action = action

        next_state.g = self.g + 1
        next_state.h = next_state.get_heuristic()
        next_state.f = next_state.g + next_state.h

        return next_state

    def get_heuristic(self):
        return len(self.goal_state_positions) - self.__count_goals()

    def is_applicable(self, action: Action) -> bool:
        agent_row = self.agent_row
        agent_col = self.agent_col

        if action.type is ActionType.NoOp:
            return True
        elif action.type is ActionType.Move:
            # check that next position is free
            next_agent_row = agent_row + action.agent_row_delta
            next_agent_col = agent_col + action.agent_col_delta
            return self.is_free(next_agent_row, next_agent_col)
        elif action.type is ActionType.Push:
            # check that next agent position is box
            next_agent_row = agent_row + action.agent_row_delta
            next_agent_col = agent_col + action.agent_col_delta
            if not self.is_box(next_agent_row, next_agent_col):
                return False
            # check that next box position is free
            next_box_row = next_agent_row + action.box_row_delta
            next_box_col = next_agent_col + action.box_col_delta
            return self.is_free(next_box_row, next_box_col)
        elif action.type is ActionType.Pull:
            # check that next agent position is free
            next_agent_row = agent_row + action.agent_row_delta
            next_agent_col = agent_col + action.agent_col_delta
            if not self.is_free(next_agent_row, next_agent_col):
                return False
            # check that box position is box
            box_row = agent_row + (action.box_row_delta * -1)
            box_col = agent_col + (action.box_col_delta * -1)
            if not self.is_box(box_row, box_col):
                return False

            return True
        else:
            return False

    def is_box(self, row, col) -> bool:
        return "A" <= self.map[row][col] <= "Z"

    def is_free(self, row, col) -> bool:
        return self.map[row][col] == " "

    def __count_goals(self):
        goal_count = 0
        for row in range(len(self.map)):
            for col in range(len(self.map[row])):
                key = str([row, col])
                if key not in self.goal_state_positions:
                    continue
                val = self.map[row][col]
                goal_count += 1 if self.goal_state_positions[key] == val else 0
        return goal_count

    def __repr__(self):
        map_s = "\n".join(["".join(row) for row in self.map])
        stats = "step: {} | agent: {},{} | box: {},{}" \
            .format(self.g, self.agent_row, self.agent_col, self.box_row(), self.box_col())
        return "{}\n{}".format(stats, map_s)

    def __hash__(self):
        if self._hash is None:
            prime = 31
            _hash = 1
            _hash = _hash * prime + hash(tuple(tuple(row) for row in self.map))
            self._hash = _hash
        return self._hash

    def __lt__(self, other: 'State'):
        return self.f < other.f

    def __eq__(self, other: 'State'):
        for i, row in enumerate(self.map):
            for j, char in enumerate(row):
                if not char == other.map[i][j]:
                    return False
        return True
