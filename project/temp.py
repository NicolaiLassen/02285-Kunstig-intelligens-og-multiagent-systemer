import os
from typing import List

import torch

LEVELS_DIR = './levels'


def normalize_dist(t):
    # Normalize  # PLZ DON'T BLOW MY GRADIENT
    return (t - t.mean()) / (t.std() + 1e-10)


def read_level_file(index: int):
    level_names = os.listdir(LEVELS_DIR)[1:]  # skip dir info file ".DS_Store"
    file_name = level_names[index % len(level_names)]
    level_file = open(os.path.join(LEVELS_DIR, file_name), 'r')
    level_file_lines = [l.strip() if l.startswith("#") else l for l in level_file.readlines()]
    level_file.close()
    return level_file_lines


def parse_level_lines(color_dict, level_lines: List[str], width=50, height=50):
    num_rows = len(level_lines)
    num_cols = len(level_lines[0])
    num_agents = len([char for char in color_dict.keys() if '0' <= char <= '9'])
    level_matrix = torch.zeros(width, height)
    agent_positions = torch.zeros(num_agents, 2)

    for row, line in enumerate(level_lines):
        for col, char in enumerate(line):
            level_matrix[row][col] = ord(char)
            if '0' <= char <= '9':
                agent_positions[int(char)] = torch.tensor([row, col])

    # normalize level matrix
    level_matrix = normalize_dist(level_matrix)

    return level_matrix, agent_positions


def load_level(index: int):
    level_lines = read_level_file(index)
    colors_index = level_lines.index("#colors")
    initial_index = level_lines.index("#initial")
    goal_index = level_lines.index("#goal")
    end_index = level_lines.index("#end")

    # parse colors
    color_dict = {}
    for line in level_lines[colors_index + 1:initial_index]:
        split = line.split(':')
        color = split[0].strip().lower()
        for e in [e.strip() for e in split[1].split(',')]:
            color_dict[e] = color

    initial_level_lines = level_lines[initial_index + 1:goal_index]
    initial_level_matrix, initial_agent_positions = parse_level_lines(color_dict, initial_level_lines)

    goal_level_lines = level_lines[goal_index + 1:end_index]
    goal_level_matrix, goal_agent_positions = parse_level_lines(color_dict, goal_level_lines)

    return initial_level_matrix, initial_agent_positions, goal_level_matrix, goal_agent_positions


if __name__ == '__main__':
    a1, a2, b1, b2 = load_level(21)
    print(a1)

"""
def parse_level_lines(level_lines: List[str]):
    level_line_counter = 5

    # Read color dictionary
    color_dict = {}
    line = level_lines[level_line_counter]
    while not line.startswith('#'):
        split = line.split(':')
        color = split[0].strip().lower()
        for e in [e.strip() for e in split[1].split(',')]:
            color_dict[e] = color
        level_line_counter += 1
        line = level_lines[level_line_counter]

    # parse initial state
    for line in level_lines:
        if line.startswith("#initial"):




    initial_level_lines = level_lines[level_line_counter]
    initial_state = parse_level_lines(color_dict, initial_level_lines)

    # parse goal state
    goal_level_lines = read_level_lines(level_file)
    goal_state = parse_level_lines(color_dict, goal_level_lines)

    return initial_state, goal_state
"""
