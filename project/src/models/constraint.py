from src.models.conflict import Conflict
from src.models.position import Position


class Constraint:
    def __init__(self, agent: int, position: Position, step: int, conflict: Conflict):
        self.agent: int = agent
        self.position: Position = position
        self.step: int = step
        self.conflict: Conflict = conflict

    def __repr__(self):
        return "CONSTRAINT: agent: {}, position: {}, step: {}\n" \
            .format(self.agent, self.position, self.step)
