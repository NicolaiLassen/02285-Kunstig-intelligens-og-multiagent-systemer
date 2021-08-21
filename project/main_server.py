import sys
from typing import List

from environment.action import Action
from environment.env_wrapper import CBSEnvWrapper

client_name = "46"


def get_server_lines(server_out):
    lines = []
    line = ""
    while not line.startswith("#end"):
        line = server_out.readline()
        line = line.strip().replace("\n", "") if line.startswith("#") else line.replace("\n", "")
        lines.append(line)
    return lines


def send_plan(server_out, plan: List[List[Action]]):
    # Print plan to server.
    if plan is None:
        print('Unable to solve level.', file=sys.stderr, flush=True)
        sys.exit(0)
    else:
        print('Found solution of length {}.'.format(len(plan)), file=sys.stderr, flush=True)
        for joint_action in plan:
            print("|".join(a.name_ for a in joint_action), flush=True)
            # We must read the server's response to not fill up the stdin buffer and block the server.
            response = server_out.readline()


def get_server_out():
    # Send client name to server.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding='ASCII')
    print(client_name, flush=True)

    server_out = sys.stdin
    if hasattr(server_out, "reconfigure"):
        server_out.reconfigure(encoding='ASCII')
    return server_out


if __name__ == '__main__':
    server_out = get_server_out()
    print('SearchClient', flush=True)

    lines = get_server_lines(server_out)

    env_wrapper = CBSEnvWrapper()
    env_wrapper.load(file_lines=lines)
    print(env_wrapper.goal_state_positions)
