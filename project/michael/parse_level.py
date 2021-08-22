from typing import List

from michael.level import Level


def parse_level(file_lines: List[str]):
    colors_index = file_lines.index("#colors")
    initial_index = file_lines.index("#initial")
    goal_index = file_lines.index("#goal")
    end_index = file_lines.index("#end")

    # parse colors
    color_dict: dict[str, str] = dict()
    for line in file_lines[colors_index + 1:initial_index]:
        split = line.split(':')
        color = split[0].strip().lower()
        for e in [e.strip() for e in split[1].split(',')]:
            color_dict[e] = color

    agents_n = len([char for char in color_dict.keys() if '0' <= char <= '9'])

    # parse initial level state
    initial_lines = file_lines[initial_index + 1:goal_index]
    initial_state = parse_level_lines(initial_lines)

    goal_lines = file_lines[goal_index + 1:end_index]
    goal_state = parse_level_lines(goal_lines)

    rows_n = len(initial_lines)
    cols_n = len(initial_lines[0])

    return Level(
        rows_n=rows_n,
        cols_n=cols_n,
        agents_n=agents_n,
        color_dict=color_dict,
        initial_state=initial_state,
        goal_state=goal_state,
    )


def parse_level_lines(level_lines: List[str]) -> List[List[str]]:
    num_rows = len(level_lines)
    num_cols = len(level_lines[0])
    level_matrix = [["" for _ in range(num_cols)] for _ in range(num_rows)]
    for row, line in enumerate(level_lines):
        for col, char in enumerate(line):
            level_matrix[row][col] = char
    return level_matrix
