import copy
from typing import List, Dict

from src.state import State


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
            color_dict: Dict[str, str],
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

    def get_agent_state(self, agent: str):
        agent_goal_state = self.__get_state(self.goal_state, agent)
        goals = self.__get_goals(agent_goal_state.map)
        agent_initial_state = self.__get_state(self.initial_state, agent)
        agent_initial_state.goal_state_positions = goals
        return agent_initial_state

    def __get_goals(self, map: List[List[str]]):
        goals = {}
        for row in range(len(map)):
            for col in range(len(map[0])):
                char = map[row][col]
                if '0' <= char <= '9':
                    goals[str([row, col])] = char
                if 'A' <= char <= 'Z':
                    goals[str([row, col])] = char
        return goals

    def __get_state(self, map: List[List[str]], agent: str) -> State:
        agent_map = copy.deepcopy(map)
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
                    continue
                if '0' <= char <= '9':
                    agent_map[ri][ci] = " "
                if 'A' <= char <= 'Z':
                    if not self.color_dict[char] == agent_color:
                        agent_map[ri][ci] = " "
        return State(
            map=agent_map,
            agent=agent,
            agent_row=agent_row,
            agent_col=agent_col
        )
