import os
import sys
from typing import List
from typing import Tuple

import torch
from torch import Tensor

from agents.ppo_agent import PPOAgent
from environment.env_wrapper import EnvWrapper
from environment.level_state import LevelState
from models.curiosity import IntrinsicCuriosityModule
from models.policy_models import ActorPolicyModel, PolicyModel

LEVELS_DIR = './levels'


def read_level_file(index: int):
    level_names = os.listdir(LEVELS_DIR)[1:]  # skip dir info file ".DS_Store"
    file_name = level_names[index % len(level_names)]
    level_file = open(os.path.join(LEVELS_DIR, file_name), 'r')

    level_file_lines = [line.strip().replace("\n", "") if line.startswith("#") else line.replace("\n", "")
                        for line in level_file.readlines()]
    for line in level_file_lines:
        print(line, file=sys.stderr, flush=True)
    level_file.close()
    return level_file_lines


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


def parse_level_lines(color_dict, level_lines: List[str], width=50, height=50) -> LevelState:
    num_agents = len([char for char in color_dict.keys() if '0' <= char <= '9'])
    num_rows = len(level_lines)
    num_cols = len(level_lines[0])
    level_matrix: Tensor = torch.zeros(width, height, dtype=torch.long)
    color_matrix: Tensor = torch.zeros(width, height, dtype=torch.long)
    agent_positions = torch.zeros(num_agents, 2)

    for row, line in enumerate(level_lines):
        for col, char in enumerate(line):
            level_matrix[row][col] = ord(char)
            if '0' <= char <= '9':
                agent_positions[int(char)] = torch.tensor([row, col])
                color_matrix[row][col] = color_to_int(color_dict[char])
            if 'A' <= char <= 'Z':
                color_matrix[row][col] = color_to_int(color_dict[char])

    return LevelState(
        num_rows,
        num_cols,
        level_matrix,
        color_matrix,
        agent_positions,
    )


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



def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp

if __name__ == '__main__':
    initial_state, goal_state = load_level(1)

    width = 50
    height = 50

    lr_actor = 3e-4
    lr_critic = 1e-3
    lr_icm = 1e-3

    # print(initial_state, file=sys.stderr, flush=True)
    #
    env = EnvWrapper(
        action_space_n=29,
        initial_state=initial_state,
        goal_state=goal_state,
    )
    # print(env)
    #
    # for a in [Action.PushSS, Action.PushSS, Action.PushSS, Action.MoveN, Action.MoveN, Action.MoveN, Action.MoveE,
    #           Action.MoveE, Action.MoveE]:
    #     s1, r, d = env.step([a])
    #     print(env)
    #     print(d)

    actor = ActorPolicyModel(width, height, env.action_space_n).cuda()
    critic = PolicyModel(width, height).cuda()
    icm = IntrinsicCuriosityModule(env.action_space_n).cuda()

    optimizer = torch.optim.Adam([
        {'params': actor.parameters(), 'lr': lr_actor},
        {'params': icm.parameters(), 'lr': lr_icm},
        {'params': critic.parameters(), 'lr': lr_critic}
    ])

    print(get_n_params(actor))
    print(get_n_params(critic))
    agent = PPOAgent(env, actor, critic, icm, optimizer)
    agent.train(2000, 10000)
