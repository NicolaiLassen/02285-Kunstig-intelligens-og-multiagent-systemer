import sys
from queue import PriorityQueue
from typing import List, Dict

from environment.action import Action, ActionType
from environment.env_wrapper import CBSEnvWrapper
from environment.state import Constraint, State
from server import get_server_out, get_server_lines, send_plan
from utils.frontier import FrontierBestFirst


class CTNode:
    constraints: [Constraint] = []
    solutions: Dict[int, List[State]] = {}
    cost: int

    def __init__(self, constraints=None, solutions=None, cost=None):
        self.constraints = constraints
        self.solutions = solutions
        self.cost = cost


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


def extract_plan(sol: State) -> '[Action, ...]':
    plan = [None for _ in range(sol.g)]
    state = sol
    while state.action is not None:
        plan[state.g - 1] = state.action
        state = state.parent
    return plan


def merge_paths(node: CTNode):
    path_dict = {}

    for i in range(len(node.solutions)):
        path_dict[i] = extract_plan(node.solutions[i][-1])

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


def get_low_level_plan(lines, index, constraints):
    frontier = FrontierBestFirst()
    env_wrapper2 = CBSEnvWrapper()
    s = env_wrapper2.load(lines, index)

    frontier.add(s)

    explored = set()

    while True:
        if frontier.is_empty():
            break

        node = frontier.pop()

        if node.is_goal_state():
            return node.extract_plan()

        explored.add(node)

        for state in node.get_expanded_states(index, constraints):
            is_not_frontier = not frontier.contains(state)
            is_explored = state not in explored
            if is_not_frontier and is_explored:
                frontier.add(state)


def sic(path_dict):
    count = 0
    for agent_path in path_dict.values():
        count += len(agent_path)

    return count


def tplusone(step):
    return step[0] + 1, step[1], step[2]


def get_conflicts(node: CTNode):
    num_agents = len(node.solutions)

    conflicts = []
    for step in range(get_max_path_len(node.solutions)):

        next_agent_rows = [-1 for _ in range(num_agents)]
        next_agent_cols = [-1 for _ in range(num_agents)]
        box_rows = [-1 for _ in range(num_agents)]
        box_cols = [-1 for _ in range(num_agents)]

        for a in range(num_agents):
            if step >= len(node.solutions[a]):
                continue

            next_agent_rows[a] = node.solutions[a][step].agents[a][0]
            next_agent_cols[a] = node.solutions[a][step].agents[a][1]

        for a1 in range(num_agents):
            if step >= len(node.solutions[a1]):
                continue
            if node.solutions[a1][step].action is ActionType.NoOp:
                continue
            for a2 in range(num_agents):
                if step >= len(node.solutions[a2]):
                    continue
                if a1 == a2:
                    continue
                if node.solutions[a2][step] is ActionType.NoOp:
                    continue

                # is moving same box
                if box_rows[a1] == box_rows[a2] and box_cols[a1] == box_cols[a2]:
                    conflicts.append(Constraint(a1, node.solutions[a1][step], step, [a1, a2]))
                    conflicts.append(Constraint(a2, node.solutions[a2][step], step, [a1, a2]))
                    break

                # is moving into same position
                if next_agent_rows[a1] == next_agent_rows[a2] and next_agent_cols[a1] == next_agent_cols[a2]:
                    conflicts.append(Constraint(a1, node.solutions[a1][step], step, [a1, a2]))
                    conflicts.append(Constraint(a2, node.solutions[a2][step], step, [a1, a2]))
                    break

    return conflicts


if __name__ == '__main__':
    server_out = get_server_out()

    # Send client name to server.
    print('SearchClient', flush=True)

    # Read level lines from server
    lines = get_server_lines(server_out)

    # Parse level lines
    # agents_n, initial_state, goal_state = load_level(lines)

    env_wrapper = CBSEnvWrapper()
    s = env_wrapper.load(lines, 0)

    path_dict = {}

    for i in range(len(s.agents)):
        plan = get_low_level_plan(lines, i, [])
        path_dict[i] = plan

    open = [CTNode(
        constraints=[],
        solutions=path_dict,
        cost=sic(path_dict)
    )]

    while len(open) > 0:
        p = open.pop()

        conflicts = get_conflicts(p)
        if len(conflicts) == 0:
            send_plan(server_out, merge_paths(p))
            break

        c = conflicts.pop()
        print("conflicts.agents: {}".format(c.agents), file=sys.stderr)
        for a in c.agents:
            node = CTNode()
            node.constraints = p.constraints
            node.constraints.append(c)
            solutions = p.solutions
            solutions[a] = get_low_level_plan(lines, a, constraints=node.constraints)
            node.solutions = solutions
            node.cost = sic(solutions)
            open.append(node)

# env_wrapper = CBSEnvWrapper()
# env_wrapper.load(file_lines=lines)
# print(env_wrapper.goal_state_positions)
