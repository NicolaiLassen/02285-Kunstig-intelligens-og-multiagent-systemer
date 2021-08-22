import sys

from michael.level import Level
from michael.parse_level import parse_level
from michael.server import get_server_out, get_server_lines


def get_agent_map(agent: str, level: Level):
    map = level.initial_state.copy()
    agent_color = level.color_dict[agent]
    for ri, row in enumerate(map):
        for ci, char in enumerate(row):
            if char == "+" or char == " " or char == agent:
                continue
            if '0' <= char <= '9':
                map[ri][ci] = "+"
            if 'A' <= char <= 'Z':
                if not level.color_dict[char] == agent_color:
                    map[ri][ci] = "+"
    return map


def expand_state(state):
    # NoOp
    state.g += 1
    nodes = [state]

    return nodes


if __name__ == '__main__':
    server_out = get_server_out()

    # Send client name to server.
    print('SearchClient', flush=True)

    # Read level lines from server
    lines = get_server_lines(server_out)

    # Parse level lines
    level = parse_level(lines)

    # Create agent map
    agent_map = get_agent_map("0", level)

    print(agent_map, flush=True, file=sys.stderr)
