import sys

import torch
import torch.nn as nn

from agents.ppo_agent import PPOAgent
from environment.action import Action, action_dict
from environment.env_wrapper import EnvWrapper
from models.policy_models import PolicyModel, PolicyModelEncoder
from utils.preprocess import parse_level_file


def debug_print(message):
    print(message, file=sys.stderr, flush=True)


if __name__ == '__main__':

    # Send client name to server.

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding='ASCII')
    print('SearchClient', flush=True)

    server_messages = sys.stdin
    if hasattr(server_messages, "reconfigure"):
        server_messages.reconfigure(encoding='ASCII')

    bach_size = 1
    width = 50
    height = 50

    lr_actor = 0.0005
    lr_critic = 0.001

    initial_state, goal_state = parse_level_file(server_messages)

    # PRINT STUFF
    debug_print(initial_state.level_matrix)
    debug_print(initial_state.box_places)

    action_space_n = int(len(action_dict))
    env_wrapper = EnvWrapper(
        initial_state,
        goal_state,
        action_space_n,
        initial_state.level_matrix,
        initial_state.color_matrix,
        initial_state.agent_places,
        initial_state.box_places,
        goal_state.level_matrix,
        nn.Linear(200, 200),
        initial_state.mask
    )

    debug_print(env_wrapper.action_space_n)
    actor = PolicyModelEncoder(width, height, env_wrapper.action_space_n)
    critic = PolicyModel(width, height)

    optimizer = torch.optim.Adam([
        {'params': actor.parameters(), 'lr': lr_actor},
        {'params': critic.parameters(), 'lr': lr_critic}
    ])

    agent = PPOAgent(env_wrapper, actor, critic, optimizer)
    agent.train(150, 100000)

    plan = [[Action.MoveE]]

    # Print plan to server.
    if plan is None:
        print('Unable to solve level.', file=sys.stderr, flush=True)
        sys.exit(0)
    else:
        print('Found solution of length {}.'.format(len(plan)), file=sys.stderr, flush=True)

        for joint_action in plan:
            print("|".join(a.name_ for a in joint_action), flush=True)
            # We must read the server's response to not fill up the stdin buffer and block the server.
            response = server_messages.readline()
