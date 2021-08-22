from environment.action import Action
from environment.level_loader import load_level_state
from environment.state import State
from michael.parse_level import parse_level
from server import get_server_out, get_server_lines, send_plan
from utils.frontier import FrontierBestFirst


class CTNode:
    constraints: set = set()
    solutions: dict = {}
    cost: int

    def __init__(self, constraints=None, solutions=None, cost=None):
        self.constraints = constraints
        self.solutions = solutions
        self.cost = cost


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

    merged_path = [[Action.NoOp for _ in range(agents_n)] for _ in range(max_path_len)]
    for i in range(max_path_len):
        for agent in agents:
            agent_path = path_dict[agent]
            if i >= len(agent_path):
                merged_path[i][agent] = Action.NoOp
            else:
                merged_path[i][agent] = agent_path[i]

    return merged_path


def get_low_level_plan(initial_state: State, constraints=[]):
    frontier = FrontierBestFirst()
    explored = set()

    frontier.add(initial_state)

    while not frontier.is_empty():
        node = frontier.pop()

        if node.is_goal_state():
            return node.extract_plan()

        explored.add(node)

        for state in node.get_expanded_states(constraints):
            is_not_frontier = not frontier.contains(state)
            is_explored = state not in explored
            if is_not_frontier and is_explored:
                frontier.add(state)

    return None


def sic(path_dict):
    return 0


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


if __name__ == '__main__':
    server_out = get_server_out()

    # Send client name to server.
    print('SearchClient', flush=True)

    # Read level lines from server
    lines = get_server_lines(server_out)

    # Parse level lines
    level = parse_level(lines)


    path_dict = {}

    for i in range(level.agents_n):
        s = load_level_state(lines, i)
        plan = get_low_level_plan(s)
        path_dict[i] = plan

    open = [CTNode(
        constraints=set(),
        solutions=path_dict,
        cost=sic(path_dict)
    )]

    while len(open) > 0:
        p = open.pop()

        conflicts = get_conflicts(p, 0)

        if len(conflicts) == 0:
            # return p
            exit()

        c = conflicts.pop()
        for a in c.agents:
            node = CTNode()
            # constraints=set(p.constraints, (a, state, t))
            solutions = p.solutions,
            cost = p.cost

    send_plan(server_out, merge_paths(path_dict))

# env_wrapper = CBSEnvWrapper()
# env_wrapper.load(file_lines=lines)
# print(env_wrapper.goal_state_positions)
