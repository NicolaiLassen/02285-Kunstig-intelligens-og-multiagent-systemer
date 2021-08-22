from typing import List


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
