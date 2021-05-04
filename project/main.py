import sys

import torch.nn as nn

from environment.EnvWrapper import EnvWrapper
from environment.action import Action
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

    initial_state, goal_state = parse_level_file(server_messages)
    debug_print(initial_state.level_matrix)

    envWrapper = EnvWrapper(initial_state.num_agents,
                            initial_state.level_matrix,
                            initial_state.color_matrix,
                            {},
                            goal_state.level_matrix,
                            nn.Linear(200, 200))

    plan = [[Action.PushWW]]

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
