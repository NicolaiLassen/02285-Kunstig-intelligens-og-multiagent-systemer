from queue import PriorityQueue

from environment.env_wrapper import CBSEnvWrapper
from server import get_server_out, get_server_lines, send_plan
from utils.frontier import FrontierBestFirst


def get_astar_path(n, initial_state, goal_state):
    q = PriorityQueue()
    visited = set()

    q.put((0, initial_state))

    return [1, 2, 3, 4]


def get_max_path_len(path_dict):
    max_path_len = 0
    for agent in path_dict.keys():
        path_len = len(path_dict[agent])
        max_path_len = path_len if path_len >= max_path_len else max_path_len
    return max_path_len


def merge_paths(path_dict):
    max_path_len = get_max_path_len(path_dict)
    agents = path_dict.keys()
    agents_n = len(agents)

    merged_path = [[None for _ in range(agents_n)] for _ in range(max_path_len)]
    for i in range(max_path_len):
        for agent in agents:
            agent_path = path_dict[agent]
            merged_path[i][agent] = agent_path[i]

    return merged_path


if __name__ == '__main__':
    server_out = get_server_out()

    # Send client name to server.
    print('SearchClient', flush=True)

    # Read level lines from server
    lines = get_server_lines(server_out)

    # Parse level lines
    # agents_n, initial_state, goal_state = load_level(lines)

    env_wrapper = CBSEnvWrapper()
    s = env_wrapper.load(lines, "test")

    frontier = FrontierBestFirst()

    frontier.add(s)

    explored = set()

    while True:
        if frontier.is_empty():
            break

        node = frontier.pop()

        if node.is_goal_state():
            send_plan(server_out, [ [i] for i in node.extract_plan()])
            break

        explored.add(node)

        for state in node.get_expanded_states():
            is_not_frontier = not frontier.contains(state)
            is_explored = state not in explored
            if is_not_frontier and is_explored:
                frontier.add(state)

    # # Find a-star plan for each agent
    # path_dict = {}
    # for n in range(agents_n):
    #     agent_plan = get_astar_path(n, initial_state, goal_state)
    #     path_dict[n] = agent_plan
    #
    # master_plan = merge_paths(path_dict)
    #
    # print("master_plan", flush=True, file=sys.stderr)
    # print(master_plan, flush=True, file=sys.stderr)


def tplusone(step):
    return step[0] + 1, step[1], step[2]


def get_conflicts(agents: [], path_dict, conflicts=None):
    if not bool(conflicts):
        conflicts_db = dict()

    # random.shuffle(agents)

    for agent in agents:
        if agent not in conflicts:
            conflicts[agent] = set()

        if path_dict[agent]:
            agent_path = path_dict[agent]
            agent_path_len = len(agent_path)

    return conflicts

# env_wrapper = CBSEnvWrapper()
# env_wrapper.load(file_lines=lines)
# print(env_wrapper.goal_state_positions)
