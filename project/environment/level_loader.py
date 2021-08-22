from typing import List, Tuple

import numpy as np
import torch

from environment.level_state import TempState


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
    agent_positions = np.zeros((num_agents, 2))

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
