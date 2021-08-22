import sys
from typing import List, Dict

from environment.action import Action, ActionType
from michael.a_state import AState, Constraint
from michael.parse_level import parse_level
from server import get_server_out, get_server_lines, send_plan
from utils.frontier import FrontierBestFirst


## TODO ######!!!!!!!!!!!!!!!
## Fix hvis agenten ikke kan finde en vej til at starte med pga blok

class Conflict:
    def __init__(self, agents, states, t):
        self.agents: [int] = agents
        self.states: Dict[str, AState] = states
        self.t = t


class CTNode:
    constraints: List[Constraint] = []
    solutions: Dict[int, List[AState]] = {}
    solution_nodes: Dict[int, List[AState]] = {}
    cost: int

    def __init__(self, constraints=None, solutions=None, solution_nodes=None, cost=None):
        self.constraints = constraints
        self.solutions = solutions
        self.solution_nodes = solution_nodes
        self.cost = cost


def check_conflict(node: CTNode) -> Conflict:
    num_agents = len(node.solution_nodes)
    for step in range(get_max_path_len(node.solution_nodes)):

        next_agent_rows = [-1 for _ in range(num_agents)]
        next_agent_cols = [-1 for _ in range(num_agents)]
        box_rows = [-1 for _ in range(num_agents)]
        box_cols = [-1 for _ in range(num_agents)]

        for a in range(num_agents):
            if step >= len(node.solution_nodes[a]):
                continue

            next_agent_rows[a] = node.solution_nodes[a][step].agent_row
            next_agent_cols[a] = node.solution_nodes[a][step].agent_col

        for a1 in range(num_agents):
            if step >= len(node.solution_nodes[a1]):
                continue
            if node.solution_nodes[a1][step].action is ActionType.NoOp:
                continue

            for a2 in range(num_agents):
                if step >= len(node.solution_nodes[a2]):
                    continue
                if a1 == a2:
                    continue
                if node.solution_nodes[a2][step].action is ActionType.NoOp:
                    continue

                # is moving same box
                # if box_rows[a1] == box_rows[a2] and box_cols[a1] == box_cols[a2]:
                #     return Conflict([a1, a2], node.solution_nodes[a1][step].g)

                # is moving into same position
                if next_agent_rows[a1] == next_agent_rows[a2] and next_agent_cols[a1] == next_agent_cols[a2]:
                    temp_fuck_states = {
                        str(a1): node.solution_nodes[a1][step],
                        str(a2): node.solution_nodes[a2][step]
                    }
                    return Conflict([str(a1), str(a2)], temp_fuck_states, node.solution_nodes[a1][step].g)
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
            return state.extract_nodes(), state.extract_plan()

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

    plans = {}
    node_list = {}

    for a in range(level.agents_n):
        initial_state = level.get_agent_states(str(a))
        plan_nodes, plan = get_low_level_plan(initial_state)
        plans[a] = plan
        node_list[a] = plan_nodes

    open = [CTNode(
        constraints=[],
        solutions=plans,
        solution_nodes=node_list,
        cost=sic(plans)
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
            node.constraints.append(Constraint(a, conflict.states, conflict.t))
            node.solutions = p.solutions
            node.solution_nodes = p.solution_nodes

            plan_nodes, plan = get_low_level_plan(level.get_agent_states(str(a)), constraints=node.constraints)
            node.solution_nodes[int(a)] = plan_nodes
            node.solutions[int(a)] = plan

            open.append(node)
