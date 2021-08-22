from typing import List

from michael.level import Level


class AState:
    map: List[List[str]]
    agent_row: int
    agent_col: int
    g: int
    h: int
    f: int

    def __init__(self, map, agent_row, agent_col):
        self.map = map
        self.agent_row = agent_row
        self.agent_col = agent_col
        self.g = 0
        self.h = 0
        self.f = 0


def get_agent_state(agent: str, level: Level) -> AState:
    agent_map = level.initial_state.copy()
    agent_color = level.color_dict[agent]
    agent_row = 0
    agent_col = 0
    for ri, row in enumerate(agent_map):
        for ci, char in enumerate(row):
            if char == "+" or char == " ":
                continue
            if char == agent:
                agent_row = ri
                agent_col = ci
            if '0' <= char <= '9':
                agent_map[ri][ci] = "+"
            if 'A' <= char <= 'Z':
                if not level.color_dict[char] == agent_color:
                    agent_map[ri][ci] = "+"
    return AState(
        map=agent_map,
        agent_row=agent_row,
        agent_col=agent_col
    )
