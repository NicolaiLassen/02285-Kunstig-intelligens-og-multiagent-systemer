
class Constraint:
    def __init__(self, agent: str, position: [int, ...], step: int, conflict):
        self.agent: str = agent
        self.position: [int, ...] = position
        self.step: int = step
        self.conflict = conflict

    def __repr__(self):
        return "CONSTRAINT: agent: {}, position: {}, step: {}\n" \
            .format(self.agent, self.position, self.step)
