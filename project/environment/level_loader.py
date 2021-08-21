from typing import List
from typing import Tuple

import torch
from torch import Tensor

from environment.level_state import LevelState


def load_level(file_lines: List[str]) -> Tuple[LevelState, LevelState]:
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

    # parse initial level state
    level_initial_lines = file_lines[initial_index + 1:goal_index]
    level_initial_state = parse_level_lines(color_dict, level_initial_lines)

    # parse goal level state
    level_goal_lines = file_lines[goal_index + 1:end_index]
    level_goal_state = parse_level_lines(color_dict, level_goal_lines)

    return level_initial_state, level_goal_state


def parse_level_lines(color_dict, level_lines: List[str]) -> LevelState:
    num_agents = len([char for char in color_dict.keys() if '0' <= char <= '9'])
    num_rows = len(level_lines)
    num_cols = len(level_lines[0])
    level_matrix: Tensor = torch.zeros(num_rows, num_cols, dtype=torch.long)
    color_matrix: Tensor = torch.zeros(num_rows, num_cols, dtype=torch.long)
    agent_positions = torch.zeros(num_agents, 2)

    for row, line in enumerate(level_lines):
        for col, char in enumerate(line):
            val = ord(char)
            level_matrix[row][col] = val
            if '0' <= char <= '9':
                agent_positions[int(char)] = torch.tensor([row, col])
            if '0' <= char <= '9' or 'A' <= char <= 'Z':
                color_matrix[row][col] = color_to_int(color_dict[char])

    return LevelState(
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
