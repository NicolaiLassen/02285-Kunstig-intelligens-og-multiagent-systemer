
class Entity:
    def __init__(self, char, col: int, row: int, color):
        self.col = col
        self.row = row
        self.char = char
        self.color = color

    def is_box(self):
        return self.char == '+'

    def is_free(self):
        return self.char == ' '

    def is_agent(self):
        return '0' <= self.char <= '9'

    def is_box(self):
        return 'A' <= self.char <= 'Z'