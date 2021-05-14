# -*- coding: utf-8 -*-
import copy
import random
from random import randint

# Easy to read representation for each cardinal direction.
N, S, W, E = ('n', 's', 'w', 'e')


class Cell(object):
    """
    Class for each individual cell. Knows only its position and which walls are
    still standing.
    """

    def __init__(self, x, y, walls):
        self.x = x
        self.y = y
        self.walls = set(walls)

    def __repr__(self):
        # <15, 25 (es  )>
        return '<{}, {} ({:4})>'.format(self.x, self.y, ''.join(sorted(self.walls)))

    def __contains__(self, item):
        # N in cell
        return item in self.walls

    def is_full(self):
        """
        Returns True if all walls are still standing.
        """
        return len(self.walls) == 4

    def _wall_to(self, other):
        """
        Returns the direction to the given cell from the current one.
        Must be one cell away only.
        """
        assert abs(self.x - other.x) + abs(self.y - other.y) == 1, '{}, {}'.format(self, other)
        if other.y < self.y:
            return N
        elif other.y > self.y:
            return S
        elif other.x < self.x:
            return W
        elif other.x > self.x:
            return E
        else:
            assert False

    def connect(self, other):
        """
        Removes the wall between two adjacent cells.
        """
        other.walls.remove(other._wall_to(self))
        self.walls.remove(self._wall_to(other))


class Maze(object):
    """
    Maze class containing full board and maze generation algorithms.
    """

    # Unicode character for a wall with other walls in the given directions.
    UNICODE_BY_CONNECTIONS = {'ensw': '┼',
                              'ens': '├',
                              'enw': '┴',
                              'esw': '┬',
                              'es': '┌',
                              'en': '└',
                              'ew': '─',
                              'e': '╶',
                              'nsw': '┤',
                              'ns': '│',
                              'nw': '┘',
                              'sw': '┐',
                              's': '╷',
                              'n': '╵',
                              'w': '╴'}

    def __init__(self, width=20, height=10):
        """
        Creates a new maze with the given sizes, with all walls standing.
        """
        self.width = width
        self.height = height
        self.cells = []
        for y in range(self.height):
            for x in range(self.width):
                self.cells.append(Cell(x, y, [N, S, E, W]))

    def __getitem__(self, index):
        """
        Returns the cell at index = (x, y).
        """
        x, y = index
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.cells[x + y * self.width]
        else:
            return None

    def neighbors(self, cell):
        """
        Returns the list of neighboring cells, not counting diagonals. Cells on
        borders or corners may have less than 4 neighbors.
        """
        x = cell.x
        y = cell.y
        for new_x, new_y in [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]:
            neighbor = self[new_x, new_y]
            if neighbor is not None:
                yield neighbor

    def to_str_matrix(self):
        """
        Returns a matrix with a pretty printed visual representation of this
        maze. Example 5x5:

        OOOOOOOOOOO
        O       O O
        OOO OOO O O
        O O   O   O
        O OOO OOO O
        O   O O   O
        OOO O O OOO
        O   O O O O
        O OOO O O O
        O     O   O
        OOOOOOOOOOO
        """
        str_matrix = [['+'] * (self.width * 2 + 1)
                      for i in range(self.height * 2 + 1)]

        for cell in self.cells:
            x = cell.x * 2 + 1
            y = cell.y * 2 + 1
            str_matrix[y][x] = ' '
            if N not in cell and y > 0:
                str_matrix[y - 1][x + 0] = ' '
            if S not in cell and y + 1 < self.width:
                str_matrix[y + 1][x + 0] = ' '
            if W not in cell and x > 0:
                str_matrix[y][x - 1] = ' '
            if E not in cell and x + 1 < self.width:
                str_matrix[y][x + 1] = ' '

        return str_matrix

    def __repr__(self):
        skinny_matrix = self.to_str_matrix()
        return '\n'.join(''.join(line) for line in skinny_matrix) + '\n'

    def randomize(self):
        """
        Knocks down random walls to build a random perfect maze.

        Algorithm from http://mazeworks.com/mazegen/mazetut/index.htm
        """
        cell_stack = []
        cell = random.choice(self.cells)
        n_visited_cells = 1

        while n_visited_cells < len(self.cells):
            neighbors = [c for c in self.neighbors(cell) if c.is_full()]
            if len(neighbors):
                neighbor = random.choice(neighbors)
                cell.connect(neighbor)
                cell_stack.append(cell)
                cell = neighbor
                n_visited_cells += 1
            else:
                cell = cell_stack.pop()

    @staticmethod
    def generate(width=20, height=10):
        """
        Returns a new random perfect maze with the given sizes.
        """
        m = Maze(width, height)
        m.randomize()
        return m

    def get_random_position(self):
        return (random.randrange(0, self.width),
                random.randrange(0, self.height))


def write_level_file(index: int, color_lines: str, level_lines: str, goal_lines: str):
    file_name = "G{}.lvl".format(index)

    header_lines = "#domain\nhospital\n#levelname\nG{}\n#colors\n".format(index)
    file_content = "{}{}\n#initial\n{}\n#goal\n{}\n#end\n".format(header_lines, color_lines, level_lines, goal_lines)

    file = open("levels/{}".format(file_name), "w")
    file.write(file_content)
    file.close()


b_name_dic = {
    'A': 10,
    'B': 8,
    'C': 7,
    'D': 6,
    'E': 5,
    'F': 4,
}


def box_name():
    temp = [i * random.random() for i in b_name_dic.values()]
    largest = max(temp)
    index = temp.index(largest)
    b_name = list(b_name_dic.keys())[index]
    return b_name


def create_level(index: int):
    width = randint(3, 24)
    height = randint(3, 24)
    maze = Maze.generate(width, height)
    matrix = maze.to_str_matrix()

    agents_n = max(1, int(random.randrange(0, 10) * random.random()))
    box_n = random.randrange(0, min(20, int(width)))
    box_names = [box_name() for i in range(box_n)]

    h = len(matrix)
    w = len(matrix[0])

    for b_name in box_names:
        matrix[random.randrange(1, h - 1)][random.randrange(1, w - 1)] = "{}".format(b_name)

    for i in range(agents_n):
        matrix[random.randrange(1, h - 1)][random.randrange(1, w - 1)] = "{}".format(i)

    wall_replace_count = agents_n + box_n
    while wall_replace_count != 0:
        r = random.randrange(1, h - 1)
        c = random.randrange(1, w - 1)
        if matrix[r][c] == "+":
            matrix[r][c] = " "
            wall_replace_count -= 1

    matrix_g = [a for a in copy.deepcopy(matrix)]
    for i, row in enumerate(matrix_g):
        for j, char in enumerate(row):
            if "0" <= char <= "9":
                matrix_g[i][j] = " "
            if "A" <= char <= "Z":
                matrix_g[i][j] = " "

    for i in range(agents_n):
        r = random.randrange(1, h - 1)
        c = random.randrange(1, w - 1)
        if matrix_g[r][c] != "+":
            matrix_g[r][c] = "{}".format(i)

    for i in box_names:
        r = random.randrange(1, h - 1)
        c = random.randrange(1, w - 1)
        if matrix_g[r][c] != "+":
            matrix_g[r][c] = "{}".format(i)

    agent_names = [str(i) for i in list(range(agents_n))]
    color_lines = "blue: {}".format(", ".join(agent_names + list(set(box_names))))
    initial = '\n'.join(''.join(line) for line in matrix)
    goal = '\n'.join(''.join(line) for line in matrix_g)
    write_level_file(index, color_lines, initial, goal)


if __name__ == '__main__':

    for i in range(500):
        print(i)
        create_level(i)

"""

+++
+0+
+1+
+++

+++
+1+
+0+
+++


"""
