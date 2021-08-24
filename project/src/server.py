import sys
from typing import List

from src.models.action import Action

client_name = "46"


def get_server_out():
    # Send client name to server.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding='ASCII')
    print(client_name, flush=True)

    server_out = sys.stdin
    if hasattr(server_out, "reconfigure"):
        server_out.reconfigure(encoding='ASCII')
    return server_out


def get_server_lines(server_out):
    server_lines = []
    line = ""
    while not line.startswith("#end"):
        line = server_out.readline()
        line = line.strip().replace("\n", "") if line.startswith("#") else line.replace("\n", "")
        server_lines.append(line)
    return server_lines


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


def get_max_path_len(path_dict):
    max_path_len = 0
    for agent in path_dict.keys():
        agent_path = path_dict[agent]
        if agent_path is None:
            continue
        path_len = len(agent_path)
        max_path_len = path_len if path_len >= max_path_len else max_path_len
    return max_path_len


def get_path(solution: List[Action]):
    solution_len = len(solution)
    plan = [Action.NoOp for _ in range(solution_len)]
    if solution_len == 0:
        return plan

    state = solution[solution_len - 1]
    while state.action is not None:
        plan[state.g - 1] = state.action
        state = state.parent
    return plan


def merge_solutions(solutions):
    path_dict = {}
    for s in solutions.keys():
        path_dict[s] = get_path(solutions[s])

    max_path_len = get_max_path_len(path_dict)
    agents = path_dict.keys()
    agents_n = len(agents)

    merged_path = [[Action.NoOp for _ in range(agents_n)] for _ in range(max_path_len)]
    for i in range(max_path_len):
        for agent in agents:
            agent_path = path_dict[agent]
            if i >= len(agent_path):
                merged_path[i][agent] = Action.NoOp
            else:
                merged_path[i][agent] = agent_path[i]

    return merged_path
