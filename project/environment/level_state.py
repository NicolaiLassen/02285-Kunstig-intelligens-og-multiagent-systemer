from torch import Tensor


class LevelState:
    rows_n: int
    cols_n: int
    agents_n: int
    level: Tensor
    agents: Tensor

    def __init__(
            self,
            rows_n: int,
            cols_n: int,
            level: Tensor,
            agents: Tensor,
    ) -> None:
        self.rows_n = rows_n
        self.cols_n = cols_n
        self.agents_n = len(agents)
        self.level = level
        self.agents = agents

    def __repr__(self):
        return self.level.__repr__()

#    def print(self):
#        for row in self.level:
#            print(row, file=sys.stderr, flush=True)
#        print('', file=sys.stderr, flush=True)
