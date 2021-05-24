import torch

from environment.entity import Entity
from environment.level_state import LevelState
from utils.misc import normalize_dist


def parse_level_file(level_file):
    # Remove redundant
    level_file.readline()  # #domain
    level_file.readline()  # hospital
    level_file.readline()  # #levelname
    level_file.readline()  # <name>
    level_file.readline()  # #colors

    # Read color dictionary
    color_dict = {}
    line = level_file.readline()
    while not line.startswith('#'):
        split = line.split(':')
        color = split[0].strip().lower()
        for e in [e.strip() for e in split[1].split(',')]:
            color_dict[e] = color
        line = level_file.readline()

    # parse initial state
    initial_level_lines = read_level_lines(level_file)
    initial_state = parse_level_lines(color_dict, initial_level_lines)

    # parse goal state
    goal_level_lines = read_level_lines(level_file)
    goal_state = parse_level_lines(color_dict, goal_level_lines)

    return initial_state, goal_state


def read_level_lines(file):
    level_lines = []
    line = file.readline()
    while not line.startswith('#'):
        level_lines.append(line.strip())
        line = file.readline()
    return level_lines


def parse_level_lines(color_dict, level_lines, width=50, height=50):
    num_rows = len(level_lines)
    num_cols = len(level_lines[0])
    level = [[None for _ in range(num_cols)] for _ in range(num_rows)]
    level_t = torch.zeros(width, height)
    agents = []

    for row, line in enumerate(level_lines):
        for col, char in enumerate(line):
            level_t[row][col] = ord(char)
            if '0' <= char <= '9':
                level[row][col] = Entity(char, row, col, color_dict[char])
                agents.append(Entity(char, row, col, color_dict[char]))

            elif 'A' <= char <= 'Z':
                level[row][col] = Entity(char, row, col, color_dict[char])
            else:
                level[row][col] = Entity(char, row, col, None)

    agents_t = torch.zeros(len(agents), 2)
    for i, agent in enumerate(agents):
        agents_t[i] = torch.tensor([agent.col, agent.row])

    level_t = normalize_dist(level_t)  # NORM

    return LevelState(num_rows, num_cols, level, agents, level_t, agents_t)
