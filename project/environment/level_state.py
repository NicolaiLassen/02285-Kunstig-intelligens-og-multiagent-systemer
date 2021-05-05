import sys
from typing import List

from torch import Tensor

from environment.entity import Entity


class LevelState:
    def __init__(self,
                 rows_n,
                 cols_n,
                 level: List[List[Entity]],
                 agents: List[Entity],
                 level_t: Tensor,
                 agents_t: Tensor
                 ) -> None:
        self.rows_n = rows_n
        self.cols_n = cols_n
        self.level = level
        self.agents = agents
        self.agents_n = len(agents)

        # TODO NAMING
        self.level_t = level_t
        self.agents_t = agents_t

    def print(self):
        for row in self.level:
            print(row, file=sys.stderr, flush=True)
        print('', file=sys.stderr, flush=True)
