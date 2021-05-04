import argparse
import sys

from environment import memory
from utils.preprocess import parse_level_file


def log(message):
    print(message, file=sys.stderr, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple client based on state-space graph search.')
    parser.add_argument('--max-memory', metavar='<MB>', type=float, default=2048.0,
                        help='The maximum memory usage allowed in MB (soft limit, default 2048).')
    args = parser.parse_args()

    # Set max memory usage allowed (soft limit).
    memory.max_usage = args.max_memory

    # Use stderr to print to the console.
    print('SearchClient initializing.', file=sys.stderr, flush=True)

    # Send client name to server.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding='ASCII')
    print('SearchClient', flush=True)

    server_messages = sys.stdin
    if hasattr(server_messages, "reconfigure"):
        server_messages.reconfigure(encoding='ASCII')

    state = parse_level_file(server_messages)
    log('#This is a comment.')
    log(state.level_matrix)

