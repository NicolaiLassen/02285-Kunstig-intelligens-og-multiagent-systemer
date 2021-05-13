import os
from typing import List
from typing import Tuple

import torch
from torch import Tensor

from environment.level_state import LevelState

LEVELS_DIR = './levels'


def load_level(index: int) -> Tuple[LevelState, LevelState]:
    file_lines = read_level_file(index)
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


def read_level_file(index: int):
    level_names = os.listdir(LEVELS_DIR)  # skip dir info file ".DS_Store"
    level_names.sort()

    file_name = level_names[index % len(level_names)]
    print(file_name)
    level_file = open(os.path.join(LEVELS_DIR, file_name), 'r')

    level_file_lines = [line.strip().replace("\n", "") if line.startswith("#") else line.replace("\n", "")
                        for line in level_file.readlines()]
    level_file.close()
    return level_file_lines


def parse_level_lines(color_dict, level_lines: List[str], width=50, height=50) -> LevelState:
    num_agents = len([char for char in color_dict.keys() if '0' <= char <= '9'])
    num_rows = len(level_lines)
    num_cols = len(level_lines[0])
    level_matrix: Tensor = torch.zeros(width, height, dtype=torch.long)
    color_matrix: Tensor = torch.zeros(width, height, dtype=torch.long)
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
    'blue': 1,
    'red': 2,
    'cyan': 3,
    'purple': 4,
    'green': 5,
    'orange': 6,
    'grey': 7,
    'lightblue': 8,
    'brown': 9,
}


def color_to_int(s: str):
    return color_map[s.lower()]
