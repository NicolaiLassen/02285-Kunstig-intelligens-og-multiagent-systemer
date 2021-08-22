import copy
from typing import List, Tuple

import numpy as np
import torch

from environment.level_state import TempState
from environment.state import State


def load_level_state(file_lines: List[str], index: int) -> State:
    free_value = 32  # ord(" ")
    agent_0_value = 48  # ord("0")
    agent_9_value = 57  # ord("9")
    box_a_value = 65  # ord("A")
    box_z_value = 90  # ord("Z")

    initial_state, goal_state = load_level(file_lines)
    initial_state = initial_state
    goal_state = goal_state

    agents_n = initial_state.agents_n
    rows_n = initial_state.rows_n
    cols_n = initial_state.cols_n

    agent = initial_state.agents[index]
    agent_color = initial_state.colors[agent[0]][agent[1]]

    goal_state_positions = {}
    for row in range(len(goal_state.level)):
        for col in range(len(goal_state.level[row])):
            val = goal_state.level[row][col]
            color = goal_state.colors[row][col]
            if not agent_color == color:
                continue
            if box_a_value <= val <= box_z_value:
                goal_state_positions[str([row, col])] = val.item()
            if agent_0_value <= val <= agent_9_value:
                goal_state_positions[str([row, col])] = val.item()

    t0_state = copy.deepcopy(initial_state)
    return State(
        map=t0_state.level,
        agent=index,
        colors=t0_state.colors,
        agents=t0_state.agents,
        goal_state_positions=goal_state_positions
    )



def load_level(file_lines: List[str]) -> Tuple[TempState, TempState]:
    colors_index = file_lines.index("#colors")
    initial_index = file_lines.index("#initial")
    goal_index = file_lines.index("#goal")
    end_index = file_lines.index("#end")

    # parse colors
    color_dict = {}
    for line in file_lines[colors_index + 1:initial_index]:
        split = line.split(':')
        color = split[0].strip().lower()
        for e in [e.strip() for e in split[1].split(',')]:
            color_dict[e] = color

    agents_n = len([char for char in color_dict.keys() if '0' <= char <= '9'])

    # parse initial level state
    initial_lines = file_lines[initial_index + 1:goal_index]
    initial_state = parse_level_lines(color_dict, initial_lines)

    # parse goal level state
    goal_lines = file_lines[goal_index + 1:end_index]
    goal_state = parse_level_lines(color_dict, goal_lines)

    return initial_state, goal_state


def parse_level_lines(color_dict, level_lines: List[str]) -> TempState:
    num_agents = len([char for char in color_dict.keys() if '0' <= char <= '9'])
    num_rows = len(level_lines)
    num_cols = len(level_lines[0])
    level_matrix: np.ndarray = np.zeros((num_rows, num_cols))
    color_matrix: np.ndarray = np.zeros((num_rows, num_cols))
    agent_positions = np.zeros((num_agents, 2), dtype=np.int8)

    for row, line in enumerate(level_lines):
        for col, char in enumerate(line):
            val = ord(char)
            level_matrix[row][col] = val
            if '0' <= char <= '9':
                agent_positions[int(char)] = torch.tensor([row, col])
            if '0' <= char <= '9' or 'A' <= char <= 'Z':
                color_matrix[row][col] = color_to_int(color_dict[char])

    return TempState(
        num_rows,
        num_cols,
        level_matrix,
        color_matrix,
        agent_positions,
    )


color_map = {
    'blue': 0,
    'red': 1,
    'cyan': 2,
    'purple': 3,
    'green': 4,
    'orange': 5,
    'pink': 6,
    'grey': 7,
    'lightblue': 8,
    'brown': 9,
}


def color_to_int(s: str):
    return color_map[s.lower()]
