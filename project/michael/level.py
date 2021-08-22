from typing import List

from michael.a_state import AState


class Level:
    rows_n: int
    cols_n: int
    agents_n: int
    color_dict: dict
    initial_state: List[List[str]]
    goal_state: List[List[str]]

    def __init__(
            self,
            rows_n: int,
            cols_n: int,
            agents_n: int,
            color_dict: dict[str, str],
            initial_state: List[List[str]],
            goal_state: List[List[str]],
    ) -> None:
        self.rows_n = rows_n
        self.cols_n = cols_n
        self.agents_n = agents_n
        self.color_dict = color_dict
        self.initial_state = initial_state
        self.goal_state = goal_state

    def __repr__(self):
        return self.initial_state.__str__()

    def get_agent_states(self, agent: str):
        return self.__get_state(self.initial_state, agent), self.__get_state(self.goal_state, agent)

    def __get_state(self, map, agent: str) -> AState:
        agent_map = map.copy()
        agent_color = self.color_dict[agent]
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
                    if not self.color_dict[char] == agent_color:
                        agent_map[ri][ci] = "+"
        return AState(
            map=agent_map,
            agent=agent,
            agent_row=agent_row,
            agent_col=agent_col
        )
