import sys
import time

#import memory
from workspace.action import Action

globals().update(Action.__members__)

start_time = time.perf_counter()


def search(initial_state, frontier):
    iterations = 0
    frontier.add(initial_state)
    explored = set()

    while True:

        # print a status message every 10000 iteration
        if ++iterations % 1000 == 0:
            print_search_status(explored, frontier)

        # if the frontier is empty then return failure
        if frontier.isEmpty():
            return None

        # choose a leaf node and remove it from the frontier
        node = frontier.pop()

        # if the node contains a goal state then return the corresponding solution
        if node.isGoalState():
            print_search_status(explored, frontier)
            return node.extractPlan()

        # add the node to the explored set
        # and expand the chosen node, adding the resulting nodes to the frontier
        explored.add(node)
        for state in node.getExpandedStates():
            if not frontier.contains(state) and not explored.__contains__(state):
                frontier.add(state)


def print_search_status(explored, frontier):
    status_template = '#Expanded: {:8,}, #Frontier: {:8,}, #Generated: {:8,}, Time: {:3.3f} s\n[Alloc: {:4.2f} MB, MaxAlloc: {:4.2f} MB]'
    elapsed_time = time.perf_counter() - start_time
    #print(status_template.format(len(explored), frontier.size(), len(explored) + frontier.size(), elapsed_time,
    #                             memory.get_usage(), memory.max_usage), file=sys.stderr, flush=True)
