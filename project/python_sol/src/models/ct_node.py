from typing import List, Dict

from src.state import State, Constraint


class CTNode:
    def __init__(self, constraints: List[Constraint] = None, solutions: Dict[int, List[State]] = None,
                 cost: int = None):
        self.constraints: List[Constraint] = constraints
        self.solutions: Dict[int, List[State]] = solutions
        self.cost: int = cost

    def __lt__(self, other):
        return self.cost < other.cost

    def copy(self):
        return CTNode(
            solutions=self.solutions.copy(),
            constraints=self.constraints.copy(),
            cost=self.cost + 0,
        )
