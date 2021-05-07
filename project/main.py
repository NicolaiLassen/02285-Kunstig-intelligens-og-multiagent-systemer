import sys
from typing import List

import torch

from agents.ppo_agent import PPOAgent
from environment.action import action_dict, Action
from environment.env_wrapper import EnvWrapper
from models.policy_models import PolicyModelEncoder, PolicyModel
from utils.preprocess import parse_level_file


def send_plan(plan: List[List[Action]]):
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


def debug_print(message):
    if message is not None:
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
    action_space_n = int(len(action_dict))
    env_wrapper = EnvWrapper(len(action_dict), initial_state, goal_state)

    actor = PolicyModelEncoder(width, height, env_wrapper.action_space_n)
    critic = PolicyModel(width, height)

    optimizer_actor = torch.optim.Adam(actor.parameters(), lr=lr_actor)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=lr_critic)

    agent = PPOAgent(env_wrapper, actor, critic, optimizer_actor, optimizer_critic)
    agent.train(100, 10000)

    # action_idxs, _ = agent.act([initial_state.level_t, initial_state.agents_t])
    # plan = [idxs_to_actions(action_idxs)]
    # debug_print(plan)
    # send_plan(plan)
