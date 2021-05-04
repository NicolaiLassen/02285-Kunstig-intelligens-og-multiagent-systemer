from typing import List

from environment.entity import Entity


class LevelState:
    def __init__(self, rows_n, cols_n, matrix: List[List[Entity]], agents):
        self.rows_n = rows_n
        self.cols_n = cols_n
        self.matrix = matrix
        self.agents = agents
        self.agents_n = len(agents)
