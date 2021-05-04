from agents.AgentBase import BaseAgent
from utils import MemBuffer


# PPO Actor Critic
class PPOAgent(BaseAgent):
    mem_buffer: MemBuffer = None

    def act(self, state) -> int:
        return 0

    def train(self, max_time, max_time_steps):
        return 