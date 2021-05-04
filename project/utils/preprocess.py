from torch import Tensor


def map_to_idx() -> Tensor:
    return




def parse_level(level):
    level.readline()  # #domain
    level.readline()  # hospital
    level.readline()  # #levelname
    level.readline()  # <name>

    # Read colors.
    level.readline()  # #colors
    agent_colors = [None for _ in range(10)]
    box_colors = [None for _ in range(26)]
    line = level.readline()
    while not line.startswith('#'):
        split = line.split(':')
        color = Color.from_string(split[0].strip())
        entities = [e.strip() for e in split[1].split(',')]
        for e in entities:
            if '0' <= e <= '9':
                agent_colors[ord(e) - ord('0')] = color
            elif 'A' <= e <= 'Z':
                box_colors[ord(e) - ord('A')] = color
        line = level.readline()

    # Read initial state.
    # line is currently "#initial".
    num_rows = 0
    num_cols = 0
    level_lines = []
    line = level.readline()
    while not line.startswith('#'):
        level_lines.append(line)
        num_cols = max(num_cols, len(line))
        num_rows += 1
        line = level.readline()

    num_agents = 0
    agent_rows = [None for _ in range(10)]
    agent_cols = [None for _ in range(10)]
    walls = [[False for _ in range(num_cols)] for _ in range(num_rows)]
    boxes = [['' for _ in range(num_cols)] for _ in range(num_rows)]
    row = 0
    for line in level_lines:
        for col, c in enumerate(line):
            if '0' <= c <= '9':
                agent_rows[ord(c) - ord('0')] = row
                agent_cols[ord(c) - ord('0')] = col
                num_agents += 1
            elif 'A' <= c <= 'Z':
                boxes[row][col] = c
            elif c == '+':
                walls[row][col] = True

        row += 1
    del agent_rows[num_agents:]
    del agent_rows[num_agents:]

    # Read goal state.
    # line is currently "#goal".
    goals = [['' for _ in range(num_cols)] for _ in range(num_rows)]
    line = level.readline()
    row = 0
    while not line.startswith('#'):
        for col, c in enumerate(line):
            if '0' <= c <= '9' or 'A' <= c <= 'Z':
                goals[row][col] = c

        row += 1
        line = level.readline()

    # End.
    # line is currently "#end".

    State.agent_colors = agent_colors
    State.walls = walls
    State.box_colors = box_colors
    State.goals = goals
    return State(agent_rows, agent_cols, boxes)




