import sys
from typing import List

from environment.entity import Entity


class LevelState:
    def __init__(self, rows_n, cols_n, matrix: List[List[Entity]], agents: List[Entity]):
        self.rows_n = rows_n
        self.cols_n = cols_n
        self.matrix = matrix
        self.agents = agents
        self.agents_n = len(agents)

    def print(self):
        for row in self.matrix:
            print(row, file=sys.stderr, flush=True)
        print('', file=sys.stderr, flush=True)
