import sys
import time

import environment.memory as memory
from environment.action import Action

globals().update(Action.__members__)

start_time = time.perf_counter()


def search(initial_state, frontier):
    iterations = 0
    frontier.add(initial_state)
    explored = set()

    return [
        [Action.MoveE]
    ]


"""
    while True:

        iterations += 1
        if iterations % 1000 == 0:
            print_search_status(explored, frontier)

        if memory.get_usage() > memory.max_usage:
            print_search_status(explored, frontier)
            print('Maximum memory usage exceeded.', file=sys.stderr, flush=True)
            return None
"""


def print_search_status(explored, frontier):
    status_template = '#Expanded: {:8,}, #Frontier: {:8,}, #Generated: {:8,}, Time: {:3.3f} s\n[Alloc: {:4.2f} MB, MaxAlloc: {:4.2f} MB]'
    elapsed_time = time.perf_counter() - start_time
    print(status_template.format(len(explored), frontier.size(), len(explored) + frontier.size(), elapsed_time,
                                 memory.get_usage(), memory.max_usage), file=sys.stderr, flush=True)
