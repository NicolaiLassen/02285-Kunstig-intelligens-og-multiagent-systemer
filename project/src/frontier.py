from queue import PriorityQueue

from src.state import State


class FrontierBestFirst:

    def __init__(self):
        super().__init__()
        self.priorityQueue = PriorityQueue()
        self.set = set()

    def add(self, state: State):
        self.priorityQueue.put(state)
        self.set.add(state)

    def pop(self) -> State:
        state = self.priorityQueue.get()
        self.set.remove(state)
        return state

    def is_empty(self) -> bool:
        return self.priorityQueue.empty()

    def size(self) -> int:
        return self.priorityQueue.qsize()

    def contains(self, state) -> bool:
        return state in self.set
