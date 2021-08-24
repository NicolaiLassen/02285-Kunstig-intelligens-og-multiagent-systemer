from queue import PriorityQueue
from typing import Dict, List

from src.frontier import FrontierBestFirst
from src.models.action import ActionType
from src.models.conflict import Conflict
from src.models.constraint import Constraint
from src.models.ct_node import CTNode
from src.parse_level import parse_level
from src.server import get_server_out, get_server_lines, send_plan, merge_solutions, get_max_path_len
from src.state import State
from src.utils.log import log


def get_conflict(node: CTNode) -> Conflict:
    agents_n = len(node.solutions)
    max_solution_len = get_max_path_len(node.solutions)

    # For every step: for every agent and every other agent
    for step in range(0, max_solution_len):
        for a0 in range(agents_n):
            a0s = node.solutions[a0]

            # Skip if step is past agent solution length
            if step >= len(a0s):
                continue

            # Skip if action is NoOp
            if a0s[step].action is ActionType.NoOp:
                continue

            for a1 in range(a0 + 1, agents_n):
                a1s = node.solutions[a1]

                # Skip if agent 1 is the same as agent 2
                if a0 == a1:
                    continue

                # Skip if step is past agent solution length
                if step >= len(a1s):
                    continue

                # Skip if action is NoOp
                if a1s[step].action is ActionType.NoOp:
                    continue

                # CONFLICT if agent 1 and agent 2 is at same position
                if a0s[step].agent_row == a1s[step].agent_row and a0s[step].agent_col == a1s[step].agent_col:
                    return Conflict(
                        type='position',
                        agent_a=str(a0),
                        agent_b=str(a1),
                        position=[a0s[step].agent_row, a0s[step].agent_col],
                        step=step
                    )

                # CONFLICT if agent 1 follows agent 2
                agent_1_pos = [a0s[step].agent_row, a0s[step].agent_col]
                agent_2_last_pos = [a1s[step - 1].agent_row, a1s[step - 1].agent_col]
                if agent_1_pos == agent_2_last_pos:
                    log("agent_1_pos: {}".format(agent_1_pos))
                    log("step: {}".format(step))
                    log("a1s[step]: {}".format(a0s[step]))
                    log("a2s[step -1]: {}".format(a1s[step - 1]))
                    exit()
                    return Conflict(
                        type='follow',
                        agent_a=str(a0),  # actor/follower
                        agent_b=str(a1),  # passive/leader
                        # TODO WHY?
                        position=[a0s[step - 1].agent_row, a0s[step - 1].agent_col],
                        step=step,
                    )

                # CONFLICT if agent 2 follows agent 1
                # if a1s[step - 1].agent_row == a2s[step].agent_row and a1s[step - 1].agent_col == a2s[step].agent_col:
                #     return Conflict(
                #         type='follow',
                #         agent_a=str(a2),
                #         agent_b=str(a1),
                #         position=[a2s[step].agent_row, a2s[step].agent_col],
                #         step=step,
                #         states={
                #             str(a1): node.solutions[a1][step - 1],
                #             str(a2): node.solutions[a2][step]
                #         }
                #     )

                # is moving same box
                # if box_rows[a1] == box_rows[a2] and box_cols[a1] == box_cols[a2]:
                #     return Conflict([a1, a2], node.solution_nodes[a1][step].g)

    return None


def sic(path_dict):
    count = 0
    for agent_path in path_dict.values():
        # TODO WHAT THEN?
        if agent_path is None:
            continue

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
        # log(state)

        if state.is_goal_state():
            return state.get_solution()

        explored.add(state)

        for state in state.expand_state(constraints):
            is_not_frontier = not frontier.contains(state)
            is_explored = state not in explored
            if is_not_frontier and is_explored:
                frontier.add(state)

    # log(initial_state)
    # log(constraints)
    # log("!!!!!!!!!! NO PLAN")
    # exit("!!!!!!!!!!!!! NO PLAN")


if __name__ == '__main__':
    server_out = get_server_out()

    # Send client name to server.
    # print('SearchClient', flush=True)

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

        # log(conflict)
        if conflict is None:
            send_plan(server_out, merge_solutions(node.solutions))
            break

        for a in [conflict.agent_a, conflict.agent_b]:
            next_node = node.copy()
            # other = conflict.agent_b if a == conflict.agent_a else conflict.agent_a

            step = conflict.step if conflict.type == 'position' else conflict.step if conflict.agent_a == a else conflict.step - 1

            constraint = Constraint(a, conflict.position, step, conflict)
            next_node.constraints.append(constraint)

            # TODO use state from conflict
            solution = get_low_level_plan(level.get_agent_state(a), constraints=next_node.constraints)

            if solution is not None:
                next_node.solutions[int(a)] = solution
                next_node.cost = sic(next_node.solutions)
                open.put(next_node)

            # log(node.solutions)
            # send_plan(server_out, merge_paths(node.solutions))
            # exit()
