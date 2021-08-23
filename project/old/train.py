import argparse
import os

from old.level_loader import load_level_state
from src.frontier import FrontierBestFirst


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def absolute_file_paths(directory):
    path = os.path.abspath(directory)
    return [entry.path for entry in os.scandir(path) if entry.is_file()]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--ckpt", default="ckpt", help="ckpt name")
    args = parser.parse_args()

    create_dir('./{}'.format(args.ckpt))

    level_file_paths_man = absolute_file_paths('./levels')

    level_file = open(level_file_paths_man[1], 'r')

    level_file_lines = [line.strip().replace("\n", "") if line.startswith("#") else line.replace("\n", "")
                        for line in level_file.readlines()]
    level_file.close()

    s = load_level_state(level_file_lines, level_file_paths_man[1])

    frontier = FrontierBestFirst()

    frontier.add(s)

    explored = set()

    while True:

        if frontier.is_empty():
            print("false")
            break

        node = frontier.pop()

        if node.is_goal_state():
            print(node.extract_plan())
            break

        explored.add(node)

        for state in node.get_expanded_states(0):
            is_not_frontier = not frontier.contains(state)
            is_explored = state not in explored
            if is_not_frontier and is_explored:
                frontier.add(state)
