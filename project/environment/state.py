import random

from environment.action import Action, ActionType
from environment.color import Color


class State:
    _RNG = random.Random(1)

    @staticmethod
    def parse_level(server_messages) -> 'State':
        # We can assume that the level file is conforming to specification, since the server verifies this.
        # Read domain.
        server_messages.readline()  # #domain
        server_messages.readline()  # hospital

        # Read Level name.
        server_messages.readline()  # #levelname
        server_messages.readline()  # <name>

        # Read colors.
        server_messages.readline()  # #colors
        agent_colors = [None for _ in range(10)]
        box_colors = [None for _ in range(26)]
        line = server_messages.readline()
        while not line.startswith('#'):
            split = line.split(':')
            color = Color.from_string(split[0].strip())
            entities = [e.strip() for e in split[1].split(',')]
            for e in entities:
                if '0' <= e <= '9':
                    agent_colors[ord(e) - ord('0')] = color
                elif 'A' <= e <= 'Z':
                    box_colors[ord(e) - ord('A')] = color
            line = server_messages.readline()

        # Read initial state.
        # line is currently "#initial".
        num_rows = 0
        num_cols = 0
        level_lines = []
        line = server_messages.readline()
        while not line.startswith('#'):
            level_lines.append(line)
            num_cols = max(num_cols, len(line))
            num_rows += 1
            line = server_messages.readline()

        num_agents = 0
        agent_rows = [None for _ in range(10)]
        agent_cols = [None for _ in range(10)]
        walls = [[False for _ in range(num_cols)] for _ in range(num_rows)]
        boxes = [['' for _ in range(num_cols)] for _ in range(num_rows)]
        row = 0
        for line in level_lines:
            for col, c in enumerate(line):
                if '0' <= c <= '9':
                    agent_rows[ord(c) - ord('0')] = row
                    agent_cols[ord(c) - ord('0')] = col
                    num_agents += 1
                elif 'A' <= c <= 'Z':
                    boxes[row][col] = c
                elif c == '+':
                    walls[row][col] = True

            row += 1
        del agent_rows[num_agents:]
        del agent_rows[num_agents:]

        # Read goal state.
        # line is currently "#goal".
        goals = [['' for _ in range(num_cols)] for _ in range(num_rows)]
        line = server_messages.readline()
        row = 0
        while not line.startswith('#'):
            for col, c in enumerate(line):
                if '0' <= c <= '9' or 'A' <= c <= 'Z':
                    goals[row][col] = c

            row += 1
            line = server_messages.readline()

        # End.
        # line is currently "#end".

        State.agent_colors = agent_colors
        State.walls = walls
        State.box_colors = box_colors
        State.goals = goals
        return State(agent_rows, agent_cols, boxes)
    
    def __init__(self, agent_rows, agent_cols, boxes):
        '''
        Constructs an initial state.
        Arguments are not copied, and therefore should not be modified after being passed in.
        
        The lists walls, boxes, and goals are indexed from top-left of the level, row-major order (row, col).
               Col 0  Col 1  Col 2  Col 3
        Row 0: (0,0)  (0,1)  (0,2)  (0,3)  ...
        Row 1: (1,0)  (1,1)  (1,2)  (1,3)  ...
        Row 2: (2,0)  (2,1)  (2,2)  (2,3)  ...
        ...
        
        For example, State.walls[2] is a list of booleans for the third row.
        State.walls[row][col] is True if there's a wall at (row, col).
        
        The agent rows, columns, and colors are indexed by the agent number.
        For example, self.agent_rows[0] is the row location of agent '0'.
        
        Note: The state should be considered immutable after it has been hashed, e.g. added to a dictionary or set.
        '''
        self.agent_rows = agent_rows
        self.agent_cols = agent_cols
        self.boxes = boxes
        self.parent = None
        self.joint_action = None
        self.g = 0
        self._hash = None
    
    def apply_action(self, joint_action: '[Action, ...]') -> 'State':
        '''
        Returns the state resulting from applying joint_action in this state.
        Precondition: Joint action must be applicable and non-conflicting in this state.
        '''
        
        # Copy this state.
        copy_agent_rows = self.agent_rows[:]
        copy_agent_cols = self.agent_cols[:]
        copy_boxes = [row[:] for row in self.boxes]
        
        # Apply each action.
        for agent, action in enumerate(joint_action):
            if action.type is ActionType.NoOp:
                pass
            
            elif action.type is ActionType.Move:
                copy_agent_rows[agent] += action.agent_row_delta
                copy_agent_cols[agent] += action.agent_col_delta
            
        copy_state = State(copy_agent_rows, copy_agent_cols, copy_boxes)
        
        copy_state.parent = self
        copy_state.joint_action = joint_action[:]
        copy_state.g = self.g + 1
        
        return copy_state
    
    def is_goal_state(self) -> 'bool':
        for row in range(len(State.goals)):
            for col in range(len(State.goals[row])):
                goal = State.goals[row][col]
                
                if 'A' <= goal <= 'Z' and self.boxes[row][col] != goal:
                    return False
                elif '0' <= goal <= '9' and not (self.agent_rows[ord(goal) - ord('0')] == row and self.agent_cols[ord(goal) - ord('0')] == col):
                    return False
        return True
    
    def get_expanded_states(self) -> '[State, ...]':
        num_agents = len(self.agent_rows)
        
        # Determine list of applicable action for each individual agent.
        applicable_actions = [[action for action in Action if self.is_applicable(agent, action)] for agent in range(num_agents)]
        
        # Iterate over joint actions, check conflict and generate child states.
        joint_action = [None for _ in range(num_agents)]
        actions_permutation = [0 for _ in range(num_agents)]
        expanded_states = []
        while True:
            for agent in range(num_agents):
                joint_action[agent] = applicable_actions[agent][actions_permutation[agent]]
            
            if not self.is_conflicting(joint_action):
                expanded_states.append(self.apply_action(joint_action))
            
            # Advance permutation.
            done = False
            for agent in range(num_agents):
                if actions_permutation[agent] < len(applicable_actions[agent]) - 1:
                    actions_permutation[agent] += 1
                    break
                else:
                    actions_permutation[agent] = 0
                    if agent == num_agents - 1:
                        done = True
            
            # Last permutation?
            if done:
                break
        
        State._RNG.shuffle(expanded_states)
        return expanded_states
    
    def is_applicable(self, agent: 'int', action: 'Action') -> 'bool':
        agent_row = self.agent_rows[agent]
        agent_col = self.agent_cols[agent]
        agent_color = State.agent_colors[agent]
        
        if action.type is ActionType.NoOp:
            return True
            
        elif action.type is ActionType.Move:
            destination_row = agent_row + action.agent_row_delta
            destination_col = agent_col + action.agent_col_delta
            return self.is_free(destination_row, destination_col)
                
    def is_conflicting(self, joint_action: '[Action, ...]') -> 'bool':
        num_agents = len(self.agent_rows)
        
        destination_rows = [None for _ in range(num_agents)] # row of new cell to become occupied by action
        destination_cols = [None for _ in range(num_agents)] # column of new cell to become occupied by action
        box_rows = [None for _ in range(num_agents)] # current row of box moved by action
        box_cols = [None for _ in range(num_agents)] # current column of box moved by action
        
        # Collect cells to be occupied and boxes to be moved.
        for agent in range(num_agents):
            action = joint_action[agent]
            agent_row = self.agent_rows[agent]
            agent_col = self.agent_cols[agent]
            
            if action.type is ActionType.NoOp:
                pass
            
            elif action.type is ActionType.Move:
                destination_rows[agent] = agent_row + action.agent_row_delta
                destination_cols[agent] = agent_col + action.agent_col_delta
                box_rows[agent] = agent_row # Distinct dummy value.
                box_cols[agent] = agent_col # Distinct dummy value.
                    
        for a1 in range(num_agents):
            if joint_action[a1] is Action.NoOp:
                continue
            
            for a2 in range(a1 + 1, num_agents):
                if joint_action[a2] is Action.NoOp:
                    continue
                
                # Moving into same cell?
                if destination_rows[a1] == destination_rows[a2] and destination_cols[a1] == destination_cols[a2]:
                    return True
                        
        return False
    
    def is_free(self, row: 'int', col: 'int') -> 'bool':
        return not State.walls[row][col] and self.boxes[row][col] == '' and self.agent_at(row, col) is None
    
    def agent_at(self, row: 'int', col: 'int') -> 'char':
        for agent in range(len(self.agent_rows)):
            if self.agent_rows[agent] == row and self.agent_cols[agent] == col:
                return chr(agent + ord('0'))
        return None
    
    def extract_plan(self) -> '[Action, ...]':
        plan = [None for _ in range(self.g)]
        state = self
        while state.joint_action is not None:
            plan[state.g - 1] = state.joint_action
            state = state.parent
        return plan
    
    def __hash__(self):
        if self._hash is None:
            prime = 31
            _hash = 1
            _hash = _hash * prime + hash(tuple(self.agent_rows))
            _hash = _hash * prime + hash(tuple(self.agent_cols))
            _hash = _hash * prime + hash(tuple(State.agent_colors))
            _hash = _hash * prime + hash(tuple(tuple(row) for row in self.boxes))
            _hash = _hash * prime + hash(tuple(State.box_colors))
            _hash = _hash * prime + hash(tuple(tuple(row) for row in State.goals))
            _hash = _hash * prime + hash(tuple(tuple(row) for row in State.walls))
            self._hash = _hash
        return self._hash
    
    def __eq__(self, other):
        if self is other: return True
        if not isinstance(other, State): return False
        if self.agent_rows != other.agent_rows: return False
        if self.agent_cols != other.agent_cols: return False
        if State.agent_colors != other.agent_colors: return False
        if State.walls != other.walls: return False
        if self.boxes != other.boxes: return False
        if State.box_colors != other.box_colors: return False
        if State.goals != other.goals: return False
        return True
    
    def __repr__(self):
        lines = []
        for row in range(len(self.boxes)):
            line = []
            for col in range(len(self.boxes[row])):
                if self.boxes[row][col] != '': line.append(self.boxes[row][col])
                elif State.walls[row][col] is not None: line.append('+')
                elif self.agent_at(row, col) is not None: line.append(self.agent_at(row, col))
                else: line.append(' ')
            lines.append(''.join(line))
        return '\n'.join(lines)
