import argparse
import sys
import time

import environment.memory as memory
from environment.frontier import FrontierBFS, FrontierDFS, FrontierBestFirst
from environment.graphsearch import search
from environment.heuristic import HeuristicAStar, HeuristicWeightedAStar, HeuristicGreedy
from environment.state import State


class SearchClient:

    @staticmethod
    def print_search_status(start_time: 'int', explored: '{State, ...}', frontier: 'Frontier') -> None:
        status_template = '#Expanded: {:8,}, #Frontier: {:8,}, #Generated: {:8,}, Time: {:3.3f} s\n[Alloc: {:4.2f} MB, MaxAlloc: {:4.2f} MB]'
        elapsed_time = time.perf_counter() - start_time
        print(status_template.format(len(explored), frontier.size(), len(explored) + frontier.size(), elapsed_time,
                                     memory.get_usage(), memory.max_usage), file=sys.stderr, flush=True)

    @staticmethod
    def main(args) -> None:
        # Use stderr to print to the console.
        print('SearchClient initializing. I am sending this using the error output stream.', file=sys.stderr,
              flush=True)

        # Send client name to server.
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding='ASCII')
        print('SearchClient', flush=True)

        # We can also print comments to stdout by prefixing with a #.
        print('#This is a comment.', flush=True)

        # Parse the level.
        server_messages = sys.stdin
        if hasattr(server_messages, "reconfigure"):
            server_messages.reconfigure(encoding='ASCII')
        initial_state = State.parse_level(server_messages)

        # Select search strategy.
        frontier = None
        if args.bfs:
            frontier = FrontierBFS()
        elif args.dfs:
            frontier = FrontierDFS()
        elif args.astar:
            frontier = FrontierBestFirst(HeuristicAStar(initial_state))
        elif args.wastar is not False:
            frontier = FrontierBestFirst(HeuristicWeightedAStar(initial_state, args.wastar))
        elif args.greedy:
            frontier = FrontierBestFirst(HeuristicGreedy(initial_state))
        else:
            # Default to BFS search.
            frontier = FrontierBFS()
            print(
                'Defaulting to BFS search. Use arguments -bfs, -dfs, -astar, -wastar, or -greedy to set the search strategy.',
                file=sys.stderr, flush=True)

        # Search for a plan.
        print('Starting {}.'.format(frontier.get_name()), file=sys.stderr, flush=True)
        plan = search(initial_state, frontier)

        # Print plan to server.
        if plan is None:
            print('Unable to solve level.', file=sys.stderr, flush=True)
            sys.exit(0)
        else:
            print('Found solution of length {}.'.format(len(plan)), file=sys.stderr, flush=True)

            for joint_action in plan:
                print("|".join(a.name_ for a in joint_action), flush=True)
                # We must read the server's response to not fill up the stdin buffer and block the server.
                response = server_messages.readline()


