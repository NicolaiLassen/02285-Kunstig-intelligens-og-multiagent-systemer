from typing import Dict

from src.state import State


class Conflict:
    def __init__(self, type: str, agent_a: str, agent_b: str, states: Dict[str, State], position: [int, ...],
                 step: int):
        self.type: str = type
        self.agent_a: str = agent_a
        self.agent_b: str = agent_b
        self.states: Dict[str, State] = states
        self.position: [int, ...] = position
        self.step: int = step

    def __repr__(self):
        return 'CONFLICT: agent: {} v {}, position: {},  step: {}\n' \
            .format(self.agent_a, self.agent_b, self.position, self.step)
