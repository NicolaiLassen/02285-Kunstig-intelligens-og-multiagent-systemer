class Position:
    def __init__(self, row: int, col: int):
        self.row: int = row
        self.col: int = col

    def __eq__(self, other):
        return self.row == other.row and self.col == other.col

    def __repr__(self):
        return "{},{}".format(self.row, self.col)
