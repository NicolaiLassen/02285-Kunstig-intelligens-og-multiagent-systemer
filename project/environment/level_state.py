from torch import Tensor


class LevelState:
    rows_n: int
    cols_n: int
    agents_n: int
    level: Tensor
    colors: Tensor
    agents: Tensor

    def __init__(
            self,
            rows_n: int,
            cols_n: int,
            level: Tensor,
            colors: Tensor,
            agents: Tensor,
    ) -> None:
        self.rows_n = rows_n
        self.cols_n = cols_n
        self.agents_n = len(agents)
        self.level = level
        self.colors = colors
        self.agents = agents

    def __repr__(self):
        level_rep = self.level[:self.rows_n, :self.cols_n].__repr__()
        color_rep = self.colors[:self.rows_n, :self.cols_n].__repr__()
        return "\n".join([level_rep, color_rep])

#    def print(self):
#        for row in self.level:
#            print(row, file=sys.stderr, flush=True)
#        print('', file=sys.stderr, flush=True)
