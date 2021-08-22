def astar_search(map, start, end):
    # Create lists for open nodes and closed nodes
    frontiers = []
    explored = []

    # Create a start node and an goal node
    from michael.node import Node
    start_node = Node(start, None)
    goal_node = Node(end, None)

    # Add the start node
    frontiers.append(start_node)

    # Loop until the frontier list is empty
    while len(frontiers) > 0:

        # Sort the open list to get the node with the lowest cost first
        frontiers.sort()

        # Get the node with the lowest cost
        current_node = frontiers.pop(0)

        # Add the current node to the explored list
        explored.append(current_node)

        # Check if we have reached the goal, return the path
        if current_node == goal_node:
            path = []
            while current_node != start_node:
                # path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]

        # Unzip the current node position
        (x, y) = current_node.position

        # Get neighbors
        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]

        # Loop neighbors
        for next in neighbors:
            # Get value from map
            map_value = map.get(next)
            # Check if the node is a wall
            if (map_value == '#'):
                continue
            # Create a neighbor node
            neighbor = Node(next, current_node)
            # Check if the neighbor is in the closed list
            if (neighbor in explored):
                continue
            # Generate heuristics (Manhattan distance)
            neighbor.g = abs(neighbor.position[0] - start_node.position[0]) + abs(
                neighbor.position[1] - start_node.position[1])
            neighbor.h = abs(neighbor.position[0] - goal_node.position[0]) + abs(
                neighbor.position[1] - goal_node.position[1])
            neighbor.f = neighbor.g + neighbor.h
            # Check if neighbor is in open list and if it has a lower f value
            if (add_to_open(frontiers, neighbor) == True):
                # Everything is green, add neighbor to open list
                frontiers.append(neighbor)

    # Return None, no path is found
    return None


# Check if a neighbor should be added to open list
def add_to_open(open, neighbor):
    for node in open:
        if (neighbor == node and neighbor.f >= node.f):
            return False
    return True


# The main entry point for this module
def main():
    # Get a map (grid)
    map = {}
    chars = ['c']
    start = None
    end = None
    width = 0
    height = 0
    # Open a file
    fp = open('data\\maze.in', 'r')

    # Loop until there is no more lines
    while len(chars) > 0:
        # Get chars in a line
        chars = [str(i) for i in fp.readline().strip()]
        # Calculate the width
        width = len(chars) if width == 0 else width
        # Add chars to map
        for x in range(len(chars)):
            map[(x, height)] = chars[x]
            if (chars[x] == '@'):
                start = (x, height)
            elif (chars[x] == '$'):
                end = (x, height)

        # Increase the height of the map
        if (len(chars) > 0):
            height += 1
    # Close the file pointer
    fp.close()
    # Find the closest path from start(@) to end($)
    path = astar_search(map, start, end)
    print()
    print(path)
    print()
    draw_grid(map, width, height, spacing=1, path=path, start=start, goal=end)
    print()
    print('Steps to goal: {0}'.format(len(path)))
    print()
