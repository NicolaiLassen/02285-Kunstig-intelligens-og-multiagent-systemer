from environment.entity import Entity
from environment.level_state import LevelState


def normalize_dist(t):
    # Normalize  # PLZ DON'T BLOW MY GRADIENT
    return (t - t.mean()) / (t.std() + 1e-10)


def parse_level_file(level_file):
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


def parse_level_lines(color_dict, level_lines):
    num_rows = len(level_lines)
    num_cols = len(level_lines[0])
    num_agents = 0
    matrix = [[None for _ in range(num_cols)] for _ in range(num_rows)]
    agents = []
    for row, line in enumerate(level_lines):
        for col, char in enumerate(line):
            if '0' <= char <= '9':
                matrix[row][col] = Entity(char, row, col, color_dict[char])
                agents.append(Entity(char, row, col, color_dict[char]))
                num_agents += 1
            elif 'A' <= char <= 'Z':
                matrix[row][col] = Entity(char, row, col, color_dict[char])
            else:
                matrix[row][col] = Entity(char, row, col, None)

    return LevelState(num_rows, num_cols, matrix, agents)
