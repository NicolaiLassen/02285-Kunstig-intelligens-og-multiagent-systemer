class Conflict:
    def __init__(self, type: str, agent_a: str, agent_b: str, position: [int, ...],
                 step: int):
        self.type: str = type
        self.agent_a: str = agent_a
        self.agent_b: str = agent_b
        self.position: [int, ...] = position
        self.step: int = step

    def __repr__(self):
        return 'CONFLICT: type: {}, agent: {} v {}, position: {},  step: {}\n' \
            .format(self.type, self.agent_a, self.agent_b, self.position, self.step)
