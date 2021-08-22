class State:
    rows_n: int
    cols_n: int 
    g: int  # weight

    def __init__(
            self,
            rows_n: int,
            cols_n: int,
    ) -> None:
        self.rows_n = rows_n
        self.cols_n = cols_n

    # def __repr__(self):
    #     level_rep = self.level[:self.rows_n, :self.cols_n].__repr__()
    #     color_rep = self.colors[:self.rows_n, :self.cols_n].__repr__()
    #     return "\n".join([level_rep, color_rep])
