from copy import deepcopy
from typing import List

from src.models.action import Action, ActionType
from src.models.constraint import Constraint
from src.utils.log import log


class State:
    _hash: int = None

    goals: List  # [[char,row,col], [char,row,col]]
    action: Action = None
    parent: 'State'

    def __init__(self, map: List[List[str]], agent: str, agent_row: int, agent_col: int):
        self.map = map
        self.agent = agent
        self.agent_row = agent_row
        self.agent_col = agent_col
        self.g = 0
        self.h = 0
        self.f = self.g + self.h

    def get_solution(self) -> '[State, ...]':
        # TODO BETTER
        plan = [None for _ in range(self.g)]
        state = self
        while state.action is not None:
            plan[state.g - 1] = state
            state = state.parent
        return [state] + plan

    def expand_state(self, constraints: List[Constraint]):

        applicable_actions = [action for action in Action if self.is_applicable(action)]
        expanded_states = [self.act(action) for action in applicable_actions]

        remove_index = []
        for constraint in constraints:
            # log(constraint)
            c_row, c_col = constraint.position
            for i, state in enumerate(expanded_states):
                # log(state)
                if constraint.step == state.g:
                    if state.map[c_row][c_col] != " ":
                        remove_index.append(i)
            # log(remove_index)
            # exit()
        filtered_states = [s for i, s in enumerate(expanded_states) if i not in remove_index]

        # if len(constraints) == 1:
        #     for constraint in constraints:
        #         if constraint.step == self.g:
        #             log("!!!!!!!!!!!")
        #             log(self)
        #             log(constraint)
        #
        #             log("expandend")
        #             for s in expanded_states:
        #                 log(s)
        #
        #             log("remove_index")
        #             log(remove_index)
        #             log("filtered")
        #             for s in filtered_states:
        #                 log(s)
        #             # exit()

        return filtered_states

    def is_goal_state(self) -> bool:
        counter = 0
        for char, row, col in self.goals:
            if self.map[row][col] == char:
                counter += 1
        return counter == len(self.goals)

    def act(self, action: Action) -> 'State':
        next_state = self.copy()

        # Update agent location
        prev_agent_row = self.agent_row
        prev_agent_col = self.agent_col
        next_agent_row = prev_agent_row + action.agent_row_delta
        next_agent_col = prev_agent_col + action.agent_col_delta
        agent_value = self.map[prev_agent_row][prev_agent_col]

        next_state.agent_row = next_agent_row
        next_state.agent_col = next_agent_col

        # Update level matrices and agent pos
        # if action.type is ActionType.NoOp:
        #     return next_state
        if action.type is ActionType.Move:
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
        # greedy
        next_state.f = next_state.h
        # astar
        # next_state.f = next_state.g + next_state.h

        return next_state

    def get_heuristic(self):

        # return self.get_missing_goal_count()
        return self.get_max_manhatten_dist()

    def get_missing_goal_count(self):
        counter = 0
        for char, row, col in self.goals:
            if self.map[row][col] == char:
                counter += 1
        return len(self.goals) - counter

    def get_max_manhatten_dist(self):
        max_dist = 0
        for r, row in enumerate(self.map):
            for c, char in enumerate(row):
                for g_char, g_r, g_c in self.goals:
                    if char != g_char:
                        continue
                    dist = abs(r - g_r) + abs(c - g_c)
                    max_dist = max(max_dist, dist)
        return max_dist

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

    def box_row(self):
        return -1

    def box_col(self):
        return -1

    def __repr__(self):
        map_s = "\n".join(["".join(row) for row in self.map])
        stats = "g: {} h: {} f: {} | agent: {},{} | box: {},{}" \
            .format(self.g, self.h, self.f, self.agent_row, self.agent_col, self.box_row(), self.box_col())
        return "{}\n{}".format(stats, map_s)

    def __hash__(self):
        if self._hash is None:
            prime = 31
            _hash = 1
            _hash = _hash * prime + self.agent_row * 23 + self.agent_col * 29 * (int(self.agent))
            _hash = _hash * prime + hash(tuple(tuple(row) for row in self.map))
            self._hash = _hash
        return self._hash

    def __lt__(self, other: 'State'):
        return self.f < other.f

    def __eq__(self, other: 'State'):
        if self.g != other.g:
            return False

        for i, row in enumerate(self.map):
            for j, char in enumerate(row):
                if not char == other.map[i][j]:
                    return False
        return True

    def copy(self):
        map = [row.copy() for row in self.map]
        state = State(map, self.agent, self.agent_row, self.agent_col)
        state.goals = self.goals
        state.h = self.h
        state.g = self.g
        state.f = self.g
        return state





