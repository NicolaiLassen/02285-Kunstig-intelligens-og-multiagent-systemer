import os
from typing import List

import torch
from torch import Tensor

from environment.action import Action
from environment.env_wrapper import EnvWrapper
from environment.level_state import LevelState

LEVELS_DIR = './levels'


def read_level_file(index: int):
    level_names = os.listdir(LEVELS_DIR)[1:]  # skip dir info file ".DS_Store"
    file_name = level_names[index % len(level_names)]
    level_file = open(os.path.join(LEVELS_DIR, file_name), 'r')
    level_file_lines = [l.strip().replace("\n", "") if l.startswith("#") else l.replace("\n", "") for l in level_file.readlines()]
    level_file.close()
    return level_file_lines


def parse_level_lines(color_dict, level_lines: List[str], width=50, height=50) -> LevelState:
    num_agents = len([char for char in color_dict.keys() if '0' <= char <= '9'])
    num_rows = len(level_lines)
    num_cols = len(level_lines[0])
    level_matrix: Tensor = torch.zeros(width, height, dtype=torch.long)
    agent_positions = torch.zeros(num_agents, 2)

    for row, line in enumerate(level_lines):
        for col, char in enumerate(line):
            level_matrix[row][col] = ord(char)
            if '0' <= char <= '9':
                agent_positions[int(char)] = torch.tensor([row, col])

    return LevelState(
        num_rows,
        num_cols,
        level_matrix,
        agent_positions,
    )


def load_level(index: int) -> tuple[LevelState, LevelState]:
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

    level_goal_lines = file_lines[goal_index + 1:end_index]
    level_goal_state = parse_level_lines(color_dict, level_goal_lines)

    for l in level_initial_lines:
        print(l)

    return level_initial_state, level_goal_state


if __name__ == '__main__':

    initial_state, goal_state = load_level(0)

    env = EnvWrapper(
        action_space_n=29,
        initial_state=initial_state,
        goal_state=goal_state,
    )

    print(env)
    env.step([Action.PullNN])
    print(env)
