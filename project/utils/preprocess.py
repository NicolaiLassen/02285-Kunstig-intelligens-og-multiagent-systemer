# from torch import Tensor


# def map_to_idx() -> Tensor:
#    return
import torch

from environment.color import Color


class Entity:
    def __init__(self, type, char, color):
        self.type = type
        self.char = char
        self.color = color

    @staticmethod
    def wall():
        return Entity('wall', '+', None)

    @staticmethod
    def empty():
        return Entity('empty', ' ', None)

    def __repr__(self):
        return "{}".format(ord(self.char))


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
    state = State(initial_level_lines, color_dict)
    #    initial_level_matrix = parse_level_lines(initial_level_lines, color_dict)

    # Read goal state.
    #    goal_level_lines = read_level_lines(level_file)
    #    goal_level_matrix = parse_level_lines(initial_level_lines, color_dict)

    return state


def read_level_lines(file):
    level_lines = []
    line = file.readline()
    while not line.startswith('#'):
        level_lines.append(line.strip())
        line = file.readline()
    return level_lines


def parse_level_lines(level_lines, color_dict):
    num_rows = len(level_lines)
    num_cols = len(level_lines[0])
    level_matrix = [[Entity.empty() for _ in range(num_cols)] for _ in range(num_rows)]
    for row, line in enumerate(level_lines):
        for col, char in enumerate(line):
            if '0' <= char <= '9':
                level_matrix[row][col] = Entity('agent', char, color_dict[char])
            elif 'A' <= char <= 'Z':
                level_matrix[row][col] = Entity('box', char, color_dict[char])
            elif char == '+':
                level_matrix[row][col] = Entity.wall()
            else:
                level_matrix[row][col] = Entity.empty()
    return level_matrix


def normalize_dist(t):
    # Normalize  # PLZ DON'T BLOW MY GRADIENT
    return (t - t.mean()) / (t.std() + 1e-10)


class State:
    level_matrix = None
    color_matrix = None
    num_agents = 0

    def __init__(self, level_lines, color_dict):

        num_rows = len(level_lines)
        num_cols = len(level_lines[0])
        # [[Entity.empty() for _ in range(num_cols)] for _ in range(num_rows)]
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
        #                    level_matrix[row][col] = Entity('agent', char, color_dict[char])
        #                elif 'A' <= char <= 'Z':
        #                    level_matrix[row][col] = Entity('box', char, color_dict[char])
        #                elif char == '+':
        #                    level_matrix[row][col] = Entity.wall()
        #                else:
        #                    level_matrix[row][col] = Entity.empty()


"""
{
	type: empty|wall|agent|box
	number: 1-100
	color: ''
}
"""
