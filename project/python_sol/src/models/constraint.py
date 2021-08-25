from src.models.conflict import Conflict


class Constraint:
    def __init__(self, agent: str, position: [int, ...], step: int, conflict: Conflict):
        self.agent: str = agent
        self.position: [int, ...] = position
        self.step: int = step
        self.conflict: Conflict = conflict

    def __repr__(self):
        return "CONSTRAINT: agent: {}, position: {}, step: {}\n" \
            .format(self.agent, self.position, self.step)

    def __eq__(self, other):
        return self.step == other.step \
               and self.agent == other.agent \
               and self.position == other.position
