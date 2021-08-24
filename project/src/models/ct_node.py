from typing import List, Dict

from src.state import State, Constraint


class CTNode:
    solutions: Dict[int, List[State]] = {}
    constraints: List[Constraint] = []
    cost: int

    def __init__(self, constraints=None, solutions=None, cost=None):
        self.constraints = constraints
        self.solutions = solutions
        self.cost = cost

    def __lt__(self, other):
        return self.cost < other.cost

    def copy(self):
        return CTNode(
            solutions=self.solutions.copy(),
            constraints=self.constraints.copy(),
            cost=self.cost + 0,
        )
