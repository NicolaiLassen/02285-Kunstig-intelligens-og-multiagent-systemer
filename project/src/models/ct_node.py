from typing import List, Dict

from src.state import State, Constraint


class CTNode:
    _hash: int = None

    def __init__(self, constraints: List[Constraint] = None, solutions: Dict[int, List[State]] = None,
                 cost: int = None):
        self.constraints: List[Constraint] = constraints
        self.solutions: Dict[int, List[State]] = solutions
        self.cost: int = cost

    def __hash__(self):
        if self._hash is None:
            prime = 31
            _hash = 1
            _hash = _hash * prime + self.cost
            _hash = _hash * prime + hash(c for c in self.constraints)
            _hash = _hash * prime + hash(s for s in self.solutions)
            self._hash = _hash
        return self._hash


    def __lt__(self, other):
        return self.cost < other.cost

    def copy(self):
        return CTNode(
            solutions=self.solutions.copy(),
            constraints=self.constraints.copy(),
            cost=self.cost + 0,
        )
