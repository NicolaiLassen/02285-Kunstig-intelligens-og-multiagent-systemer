import sys

from michael.a_state import get_agent_state
from michael.expand_state import expand_state
from michael.parse_level import parse_level
from michael.server import get_server_out, get_server_lines

if __name__ == '__main__':
    server_out = get_server_out()

    # Send client name to server.
    print('SearchClient', flush=True)

    # Read level lines from server
    lines = get_server_lines(server_out)

    # Parse level lines
    level = parse_level(lines)

    # Create agent map
    agent_state = get_agent_state("0", level)

    x = expand_state(agent_state)
    print(x, flush=True, file=sys.stderr)


    print(agent_state, flush=True, file=sys.stderr)
