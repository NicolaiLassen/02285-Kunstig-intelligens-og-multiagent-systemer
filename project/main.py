import sys
from queue import PriorityQueue
from typing import Dict, List

from src.action import ActionType
from src.ct_node import CTNode
from src.frontier import FrontierBestFirst
from src.parse_level import parse_level
from src.server import get_server_out, get_server_lines, send_plan, merge_solutions, get_max_path_len
from src.state import State, Constraint, Conflict


## TODO ######!!!!!!!!!!!!!!!
## Fix hvis agenten ikke kan finde en vej til at starte med pga blok


def get_conflict(node: CTNode) -> Conflict:
    agents_n = len(node.solutions)
    max_solution_len = get_max_path_len(node.solutions)

    # For every step: for every agent and every other agent
    for step in range(max_solution_len):
        for a1 in range(agents_n):
            a1s = node.solutions[a1]

            # Skip if step is past agent solution length
            if step >= len(a1s):
                continue

            # Skip if action is NoOp
            if a1s[step].action is ActionType.NoOp:
                continue

            for a2 in range(agents_n):
                a2s = node.solutions[a2]

                # Skip if agent 1 is the same as agent 2
                if a1 == a2:
                    continue

                # Skip if step is past agent solution length
                if step >= len(a2s):
                    continue

                # Skip if action is NoOp
                if a2s[step].action is ActionType.NoOp:
                    continue

                if a1s[step].action.type is ActionType.Push:
                    log("STEP {} ------------".format(step))
                    log(a1s[step])
                    log(a2s[step])
                    log("")

                # CONFLICT if box position is the same
                if a1s[step].box_row() == a2s[step].agent_row and a1s[step].box_col() == a2s[step].agent_col:
                    return Conflict(
                        agents=[str(a1), str(a2)],
                        position=[a1s[step].agent_row, a1s[step].agent_col],
                        step=step,
                        states={
                            str(a1): node.solutions[a1][step],
                            str(a2): node.solutions[a2][step]
                        }
                    )

                # CONFLICT if agent 1 and agent 2 is at same position
                if a1s[step].agent_row == a2s[step].agent_row and a1s[step].agent_col == a2s[step].agent_col:
                    return Conflict(
                        agents=[str(a1), str(a2)],
                        position=[a1s[step].agent_row, a1s[step].agent_col],
                        step=step,
                        states={
                            str(a1): node.solutions[a1][step],
                            str(a2): node.solutions[a2][step]
                        }
                    )

                # is moving same box
                # if box_rows[a1] == box_rows[a2] and box_cols[a1] == box_cols[a2]:
                #     return Conflict([a1, a2], node.solution_nodes[a1][step].g)

    return None


def sic(path_dict):
    count = 0
    for agent_path in path_dict.values():
        count += len(agent_path)
    return count


def get_low_level_plan(initial_state: State, constraints=[]):
    frontier = FrontierBestFirst()
    explored = set()

    frontier.add(initial_state)
    while True:
        if frontier.is_empty():
            break

        state = frontier.pop()
        log(state)

        if state.is_goal_state():
            return state.get_solution()

        explored.add(state)

        for state in state.expand_state(constraints):
            is_not_frontier = not frontier.contains(state)
            is_explored = state not in explored
            if is_not_frontier and is_explored:
                frontier.add(state)


def log(s):
    print(s, flush=True, file=sys.stderr)


if __name__ == '__main__':
    server_out = get_server_out()

    # Send client name to server.
    print('SearchClient', flush=True)

    # Read level lines from server
    lines = get_server_lines(server_out)

    # Parse level lines
    level = parse_level(lines)

    # Find low level plan for all agents
    solutions: Dict[int, List[State]] = {}
    for a in range(level.agents_n):
        initial_state = level.get_agent_state(str(a))
        solutions[a] = get_low_level_plan(initial_state)

    ## Naive solution merge
    # send_plan(server_out, merge_paths(plans))


    # Conflict based search
    open = PriorityQueue()
    open.put(CTNode(
        constraints=[],
        solutions=solutions,
        cost=sic(solutions)
    ))

    while not open.empty():
        node: CTNode = open.get()
        conflict = get_conflict(node)

        log(conflict)
        if conflict is None:
            send_plan(server_out, merge_solutions(node.solutions))
            break

        for a in conflict.agents:
            next_node = node.copy()
            other = [e for e in conflict.agents if e != a][0]
            next_node.constraints.append(Constraint(a, conflict.states[a], conflict.states[other], conflict.step))
            solution = get_low_level_plan(level.get_agent_state(a), constraints=next_node.constraints)
            next_node.solutions[int(a)] = solution
            next_node.cost = sic(next_node.solutions)
            open.put(next_node)

            # log(node.solutions)
            # send_plan(server_out, merge_paths(node.solutions))
            # exit()
