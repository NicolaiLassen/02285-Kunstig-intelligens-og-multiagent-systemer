import torch

from environment.color import Color


def normalize_dist(t):
    # Normalize  # PLZ DON'T BLOW MY GRADIENT
    return (t - t.mean()) / (t.std() + 1e-10)


class Entity:
    def __init__(self, col: int, row: int, type, char, color):
        self.col = col
        self.row = row
        self.type = type
        self.char = char
        self.color = color


def parse_level_file(level_file):
    level_file.readline()  # #domain
    level_file.readline()  # hospital
    level_file.readline()  # #levelname
    level_file.readline()  # <name>
    level_file.readline()  # #colors

    color_dict = {}
    line = level_file.readline()
    while not line.startswith('#'):
        split = line.split(':')
        color = split[0].strip().lower()
        for e in [e.strip() for e in split[1].split(',')]:
            color_dict[e] = color
        line = level_file.readline()

    # Read initial state.
    initial_level_lines = read_level_lines(level_file)
    initial_state = State(initial_level_lines, color_dict)

    # Read goal state.
    goal_level_lines = read_level_lines(level_file)
    goal_state = State(goal_level_lines, color_dict)

    return initial_state, goal_state


def read_level_lines(file):
    level_lines = []
    line = file.readline()
    while not line.startswith('#'):
        level_lines.append(line.strip())
        line = file.readline()
    return level_lines


class State:
    def __init__(self, level_lines, color_dict):
        self.num_rows = len(level_lines)
        self.num_cols = len(level_lines[0])
        self.num_agents = 0
        self.agent_places = []
        self.box_places = []

        # max lvl size 50x50
        self.level_matrix = torch.zeros(50, 50, dtype=torch.float)
        self.color_matrix = torch.zeros(50, 50, dtype=torch.float)
        for row, line in enumerate(level_lines):
            for col, char in enumerate(line):
                self.level_matrix[row][col] = ord(char)
                if '0' <= char <= '9' or 'A' <= char <= 'Z':
                    self.color_matrix[row][col] = Color.from_string(color_dict[char]).value
                else:
                    self.color_matrix[row][col] = 0

                if '0' <= char <= '9':
                    self.num_agents += 1
                    self.agent_places.append(Entity(col, row, 'box', char, color_dict[char]))
                elif 'A' <= char <= 'Z':
                    self.box_places[char].append(Entity(col, row, 'box', char, color_dict[char]))
