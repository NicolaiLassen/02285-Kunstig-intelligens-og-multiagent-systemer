from typing import Dict

from src.models.position import Position
from src.state import State


class Conflict:
    def __init__(self, type: str, agent_a: int, agent_b: int, states: Dict[int, State], position: Position, step):
        self.type: str = type
        self.agent_a: int = agent_a
        self.agent_b: int = agent_b
        self.states = states
        self.position = position
        self.step = step

    def __repr__(self):
        return 'CONFLICT: agent: {} v {}, position: {},  step: {}\n' \
            .format(self.agent_a, self.agent_b, self.position, self.step)
