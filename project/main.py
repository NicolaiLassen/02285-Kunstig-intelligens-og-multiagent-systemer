from queue import PriorityQueue
from typing import Dict, List

from src.frontier import FrontierBestFirst
from src.models.action import Action
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

    # Ensure all solution lengths are the same
    for s in node.solutions.values():
        s_len = len(s)
        s_len_diff = max_solution_len - s_len
        for i in range(s_len_diff):
            s.append(s[-1].act(Action.NoOp))

    # For every step: for every agent and every other agent
    for step in range(1, max_solution_len):
        for a0 in range(agents_n):
            for a1 in range(a0 + 1, agents_n):

                a0s = node.solutions[a0]
                a1s = node.solutions[a1]

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
                agent_0_pos = [a0s[step].agent_row, a0s[step].agent_col]
                agent_1_pos_prev = [a1s[step - 1].agent_row, a1s[step - 1].agent_col]
                if agent_0_pos == agent_1_pos_prev:
                    return Conflict(
                        type='follow',
                        agent_a=str(a0),  # actor/follower
                        agent_b=str(a1),  # passive/leader
                        position=agent_0_pos,
                        step=step,
                    )

                # CONFLICT if agent 2 follows agent 1
                agent_1_pos = [a1s[step].agent_row, a1s[step].agent_col]
                agent_0_pos_prev = [a0s[step - 1].agent_row, a0s[step - 1].agent_col]
                if agent_1_pos == agent_0_pos_prev:
                    return Conflict(
                        type='follow',
                        agent_a=str(a1),  # actor/follower
                        agent_b=str(a0),  # passive/leader
                        position=agent_1_pos,
                        step=step,
                    )

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


def get_constraint(agent, conflict):
    if conflict.type == 'position':
        return Constraint(agent, conflict.position, conflict.step, conflict)

    # agent is follower
    if conflict.type == 'follow' and agent == conflict.agent_a:
        return Constraint(agent, conflict.position, conflict.step, conflict)

    # agent is leader
    if conflict.type == 'follow' and agent == conflict.agent_b:
        return Constraint(agent, conflict.position, conflict.step - 1, conflict)

    return None


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
    # send_plan(server_out, merge_solutions(solutions))
    # exit()

    # Conflict based search
    open = PriorityQueue()
    explored = set()

    root = CTNode(
        constraints=[],
        solutions=solutions,
        cost=sic(solutions)
    )

    open.put(root)

    while not open.empty():
        log('open.qsize(): {}'.format(open.qsize()))

        node: CTNode = open.get()

        conflict = get_conflict(node)

        # log(conflict)
        if conflict is None:
            send_plan(server_out, merge_solutions(node.solutions))
            break

        for a in [conflict.agent_a, conflict.agent_b]:
            next_node = node.copy()
            # other = conflict.agent_b if a == conflict.agent_a else conflict.agent_a

            constraint = get_constraint(a, conflict)
            # log('constraint: {}'.format(constraint))

            if constraint is None:
                exit("NULL constraint")

            # skip already explored constraints
            if constraint in next_node.constraints:
                continue

            next_node.constraints.append(constraint)

            # handle corridor
            # if len(node.constraints) != 0:
            #     last_constraint = node.constraints[-1]
            #     if constraint.position == last_constraint.position:
            #         if constraint.step - 1 == last_constraint.step:
            #             corridor_step = constraint.step - 1
            #             corridor_state: State = node.solutions[int(a)][corridor_step]
            #             position = [corridor_state.agent_row, corridor_state.agent_col]
            #             no_op_constraint = Constraint(a, position, corridor_step, None)
            #             next_node.constraints.append(no_op_constraint)

            # TODO use state from conflict
            # log("AAAAAAAAAAAAAAAA")
            # log(level.get_agent_state(a))

            agent_constraints = [c for c in next_node.constraints if c.agent == a]
            solution = get_low_level_plan(level.get_agent_state(a), constraints=agent_constraints)
            # log('solution: {}'.format(solution))

            # skip node if solution is None or the same
            if solution is None or solution == node.solutions[int(a)]:
                continue

            next_node.solutions[int(a)] = solution
            next_node.cost = sic(next_node.solutions)
            open.put(next_node)

            # log(node.solutions)
            # send_plan(server_out, merge_paths(node.solutions))
            # exit()
