from collections import deque


class FrontierDFS:
    def __init__(self):
        super().__init__()
        self.queue = deque()
        self.set = set()

    def add(self, state):
        self.queue.appendleft(state)
        self.set.add(state)

    def pop(self):
        state = self.queue.pop()
        self.set.remove(state)
        return state

    def is_empty(self):
        return len(self.queue) == 0

    def size(self) -> int:
        return len(self.queue)

    def contains(self, state) -> bool:
        return state in self.set
