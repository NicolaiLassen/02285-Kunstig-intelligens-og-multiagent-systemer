import copy
import sys
from abc import ABC
from typing import List

import gym as gym

from environment.action import Action, ActionType
from environment.level_loader import load_level
from environment.state import State


class Node():
    constraints = []
    cost = 0
    solution = []


class CBSEnvWrapper(gym.Env, ABC):
    initial_state = None
    goal_state = None
    goal_state_positions = {}
    t0_state = None

    n_solutions = []
    open = []

    agents_n = 0
    rows_n = 0
    cols_n = 0
    file_name = None

    def __init__(
            self
    ) -> None:
        self.free_value = 32  # ord(" ")
        self.agent_0_value = 48  # ord("0")
        self.agent_9_value = 57  # ord("9")
        self.box_a_value = 65  # ord("A")
        self.box_z_value = 90  # ord("Z")

        self.action_space_n = 29  # max actions
        self.max_agents_n = 10  # max agents

    def __repr__(self):
        return self.t0_state.__repr__()

    def low_level(self, agent: int):
        # find sol for single agent
        print()

    # def high_level(self):
    #     print()
    #     r_solutions = []
    #     for a in range(self.agents_n):
    #         sol = self.low_level(a)
    #         r_solutions.append(sol)
    #
    #     while len(self.open) != 0:
    #         print()

    # TODO: CONFLICT SHOULD BE HANDLED DIFFERENTLY
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
                    return True
                # is moving into same position
                if next_agent_rows[a1] == next_agent_rows[a2] and next_agent_cols[a1] == next_agent_cols[a2]:
                    return True

        return False

    def load(self, file_lines: List[str], index: int) -> State:
        initial_state, goal_state = load_level(file_lines)
        self.initial_state = initial_state
        self.goal_state = goal_state

        self.agents_n = initial_state.agents_n
        self.rows_n = initial_state.rows_n
        self.cols_n = initial_state.cols_n

        agent = self.initial_state.agents[index]
        print('agent: {}'.format(agent), file=sys.stderr)
        agent_color = self.initial_state.colors[agent[0]][agent[1]]

        self.goal_state_positions = {}
        for row in range(len(self.goal_state.level)):
            for col in range(len(self.goal_state.level[row])):
                val = goal_state.level[row][col]
                color = self.goal_state.colors[row][col]
                if not agent_color == color:
                    continue
                if self.box_a_value <= val <= self.box_z_value:
                    self.goal_state_positions[str([row, col])] = val.item()
                if self.agent_0_value <= val <= self.agent_9_value:
                    self.goal_state_positions[str([row, col])] = val.item()

        # free_value = 32  # ord(" ")
        # agent_0_value = 48  # ord("0")
        # agent_9_value = 57  # ord("9")
        # box_a_value = 65  # ord("A")
        # box_z_value = 90  # ord("Z")

        self.t0_state = copy.deepcopy(initial_state)
        return State(map=self.t0_state.level, colors=self.t0_state.colors, agents=self.t0_state.agents,
                     goal_state_positions=self.goal_state_positions)
