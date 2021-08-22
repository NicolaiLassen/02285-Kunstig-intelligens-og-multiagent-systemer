from typing import List

from environment.action import Action
from michael.a_state import AState
from michael.parse_level import parse_level
from server import get_server_out, get_server_lines, send_plan
from utils.frontier import FrontierBestFirst


class Conflict:
    def __init__(self, agents, state, t):
        self.agents: [int] = agents
        self.state: AState = state
        self.t = t


class Constraint:
    def __init__(self, agent, state, t):
        self.agent: int = agent
        self.state: AState = state
        self.t = t


class CTNode:
    constraints: List[Constraint] = []
    solutions: dict = {}
    cost: int

    def __init__(self, constraints=None, solutions=None, cost=None):
        self.constraints = constraints
        self.solutions = solutions
        self.cost = cost


def check_conflict(p: CTNode) -> Conflict:
    return None


def sic(path_dict):
    count = 0
    for agent_path in path_dict.values():
        count += len(agent_path)
    return count


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


def get_low_level_plan(initial_state: AState, constraints=[]):
    frontier = FrontierBestFirst()
    explored = set()

    frontier.add(initial_state)
    while True:
        if frontier.is_empty():
            break

        state = frontier.pop()

        if state.is_goal_state():
            return state.extract_plan()

        explored.add(state)

        for state in state.expand_state(constraints):
            is_not_frontier = not frontier.contains(state)
            is_explored = state not in explored
            if is_not_frontier and is_explored:
                frontier.add(state)


if __name__ == '__main__':
    server_out = get_server_out()

    # Send client name to server.
    print('SearchClient', flush=True)

    # Read level lines from server
    lines = get_server_lines(server_out)

    # Parse level lines
    level = parse_level(lines)

    solutions = {}

    for a in range(level.agents_n):
        initial_state = level.get_agent_states(str(a))
        plan = get_low_level_plan(initial_state)
        solutions[a] = plan

    open = [CTNode(
        constraints=[],
        solutions=solutions,
        cost=sic(solutions)
    )]

    while len(open) > 0:
        p = open.pop()

        conflict = check_conflict(p)
        if conflict is None:
            send_plan(server_out, merge_paths(p.solutions))
            break

        for a in conflict.agents:
            node = CTNode()
            node.constraints = p.constraints
            node.constraints.append(Constraint(a, conflict.state, conflict.t))
            solutions = p.solutions
            solutions[a] = get_low_level_plan(level.get_agent_states(str(a)), constraints=node.constraints)
            node.solutions = solutions
            node.cost = sic(solutions)
            open.append(node)
