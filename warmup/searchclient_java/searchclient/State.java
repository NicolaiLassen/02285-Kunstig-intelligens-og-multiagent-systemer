package searchclient;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;

public class State {
    private static final Random RNG = new Random(1);

    /*
        The agent rows, columns, and colors are indexed by the agent number.
        For example, this.agentRows[0] is the row location of agent '0'.
    */
    public int[] agentRows;
    public int[] agentCols;
    public static Color[] agentColors;

    /*
        The walls, boxes, and goals arrays are indexed from the top-left of the level, row-major order (row, col).
               Col 0  Col 1  Col 2  Col 3
        Row 0: (0,0)  (0,1)  (0,2)  (0,3)  ...
        Row 1: (1,0)  (1,1)  (1,2)  (1,3)  ...
        Row 2: (2,0)  (2,1)  (2,2)  (2,3)  ...
        ...

        For example, this.walls[2] is an array of booleans for the third row.
        this.walls[row][col] is true if there's a wall at (row, col).
    */
    public static boolean[][] walls;
    public char[][] boxes;
    public static char[][] goals;

    /*
        The box colors are indexed alphabetically. So this.boxColors[0] is the color of A boxes, 
        this.boxColor[1] is the color of B boxes, etc.
    */
    public static Color[] boxColors;

    public final State parent;
    public final Action[] jointAction;
    private final int g;

    private int hash = 0;


    // Constructs an initial state.
    // Arguments are not copied, and therefore should not be modified after being passed in.
    public State(int[] agentRows, int[] agentCols, Color[] agentColors, boolean[][] walls,
                 char[][] boxes, Color[] boxColors, char[][] goals
    ) {
        this.agentRows = agentRows;
        this.agentCols = agentCols;
        this.agentColors = agentColors;
        this.walls = walls;
        this.boxes = boxes;
        this.boxColors = boxColors;
        this.goals = goals;
        this.parent = null;
        this.jointAction = null;
        this.g = 0;
    }


    // Constructs the state resulting from applying jointAction in parent.
    // Precondition: Joint action must be applicable and non-conflicting in parent state.
    private State(State parent, Action[] jointAction) {


        // Copy parent
        this.agentRows = Arrays.copyOf(parent.agentRows, parent.agentRows.length);
        this.agentCols = Arrays.copyOf(parent.agentCols, parent.agentCols.length);

        this.boxes = new char[parent.boxes.length][];
        for (int i = 0; i < parent.boxes.length; i++) {
            this.boxes[i] = Arrays.copyOf(parent.boxes[i], parent.boxes[i].length);
        }

        // Set own parameters
        this.parent = parent;
        this.jointAction = Arrays.copyOf(jointAction, jointAction.length);
        this.g = parent.g + 1;

        // Apply each action
        int numAgents = this.agentRows.length;
        for (int agent = 0; agent < numAgents; ++agent) {
            Action action = jointAction[agent];
            char box;
            int boxRow;
            int boxCol;

            switch (action.type) {
                case NoOp:
                    break;
                case Move:
                    this.agentRows[agent] += action.agentRowDelta;
                    this.agentCols[agent] += action.agentColDelta;
                    break;
                case Push:
                    // <move-dir-agent> contains a box
                    boxRow = this.agentRows[agent] + action.agentRowDelta;
                    boxCol = this.agentCols[agent] + action.agentColDelta;
                    // Get char
                    box = this.boxes[boxRow][boxCol];
                    // remove box from old cell <move-dir-agent>
                    this.boxes[boxRow][boxCol] = 0;
                    // add box char to new cell <move-dir-box>
                    this.boxes[boxRow + action.boxRowDelta][boxCol + action.boxColDelta] = box;
                    // move agent to old box cell <move-dir-agent>
                    this.agentRows[agent] = boxRow;
                    this.agentCols[agent] = boxCol;
                    break;
                case Pull:
                    // opposite of <move-dir-box> contains a box
                    boxRow = this.agentRows[agent] + getOpposite(action.boxRowDelta);
                    boxCol = this.agentCols[agent] + getOpposite(action.boxColDelta);
                    // Get char
                    box = this.boxes[boxRow][boxCol];
                    // remove box from old cell opposite of <move-dir-box>
                    this.boxes[boxRow][boxCol] = 0;
                    // add box char to new cell <move-dir-box>
                    this.boxes[boxRow + action.boxRowDelta][boxCol + action.boxColDelta] = box;
                    // move agent to new cell <move-dir-agent>
                    this.agentRows[agent] += action.agentRowDelta;
                    this.agentCols[agent] += action.agentColDelta;
                    break;
            }
        }
    }

    public int g() {
        return this.g;
    }

    public boolean isGoalState() {
        for (int row = 1; row < this.goals.length - 1; row++) {
            for (int col = 1; col < this.goals[row].length - 1; col++) {
                char goal = this.goals[row][col];

                if ('A' <= goal && goal <= 'Z' && this.boxes[row][col] != goal) {
                    return false;
                } else if ('0' <= goal && goal <= '9' &&
                        !(this.agentRows[goal - '0'] == row && this.agentCols[goal - '0'] == col)) {
                    return false;
                }
            }
        }
        return true;
    }

    public ArrayList<State> getExpandedStates() {
        int numAgents = this.agentRows.length;

        // Determine list of applicable actions for each individual agent.
        Action[][] applicableActions = new Action[numAgents][];
        for (int agent = 0; agent < numAgents; ++agent) {
            ArrayList<Action> agentActions = new ArrayList<>(Action.values().length);
            for (Action action : Action.values()) {
                if (this.isApplicable(agent, action)) {
                    agentActions.add(action);
                }
            }
            applicableActions[agent] = agentActions.toArray(new Action[0]);
        }

        // Iterate over joint actions, check conflict and generate child states.
        Action[] jointAction = new Action[numAgents];
        int[] actionsPermutation = new int[numAgents];
        ArrayList<State> expandedStates = new ArrayList<>(16);
        while (true) {
            for (int agent = 0; agent < numAgents; ++agent) {
                jointAction[agent] = applicableActions[agent][actionsPermutation[agent]];
            }

            if (!this.isConflicting(jointAction)) {
                expandedStates.add(new State(this, jointAction));
            }

            // Advance permutation
            boolean done = false;
            for (int agent = 0; agent < numAgents; ++agent) {
                if (actionsPermutation[agent] < applicableActions[agent].length - 1) {
                    ++actionsPermutation[agent];
                    break;
                } else {
                    actionsPermutation[agent] = 0;
                    if (agent == numAgents - 1) {
                        done = true;
                    }
                }
            }

            // Last permutation?
            if (done) {
                break;
            }
        }

        Collections.shuffle(expandedStates, State.RNG);
        return expandedStates;
    }

    private boolean isApplicable(int agent, Action action) {
        int agentRow = this.agentRows[agent];
        int agentCol = this.agentCols[agent];
        Color agentColor = agentColors[agent];
        int boxRow;
        int boxCol;
        char box;
        int destinationRow;
        int destinationCol;
        switch (action.type) {
            case NoOp:
                return true;
            case Move:
                destinationRow = agentRow + action.agentRowDelta;
                destinationCol = agentCol + action.agentColDelta;
                return this.cellIsFree(destinationRow, destinationCol);
            case Push:
                // the neighbouring cell of the agent in direction <move-dir-agent> contains a box beta
                boxRow = agentRow + action.agentRowDelta;
                boxCol = agentCol + action.agentColDelta;
                if (this.noBoxInCell(boxRow, boxCol)) {
                    return false;
                }
                // of the same color
                box = this.boxes[boxRow][boxCol];
                if (!isBoxAndAgentSameColor(box, agentColor)) {
                    return false;
                }
                // the neighbouring cell of beta in direction <move-dir-box> is currently free.
                int boxDestinationRow = boxRow + action.boxRowDelta;
                int boxDestinationCol = boxCol + action.boxColDelta;
                return this.cellIsFree(boxDestinationRow, boxDestinationCol);
            case Pull:
                // The neighbouring cell of the agent in direction <move-dir-agent> is currently free.
                destinationRow = agentRow + action.agentRowDelta;
                destinationCol = agentCol + action.agentColDelta;
                if (!this.cellIsFree(destinationRow, destinationCol)) {
                    return false;
                }
                // The neighbouring cell of the agent in the opposite direction of <move-dir-box> contains a box beta
                boxRow = agentRow + getOpposite(action.boxRowDelta);
                boxCol = agentCol + getOpposite(action.boxColDelta);
                if (this.noBoxInCell(boxRow, boxCol)) {
                    return false;
                }
                // of the same color
                box = this.boxes[boxRow][boxCol];
                return isBoxAndAgentSameColor(box, agentColor);
        }

        // Unreachable:
        return false;
    }

    private boolean isConflicting(Action[] jointAction) {
        int numAgents = this.agentRows.length;

        int[] destinationRows = new int[numAgents]; // row of new cell to become occupied by action
        int[] destinationCols = new int[numAgents]; // column of new cell to become occupied by action
        int[] boxRows = new int[numAgents]; // current row of box moved by action
        int[] boxCols = new int[numAgents]; // current column of box moved by action

        // Collect cells to be occupied and boxes to be moved
        for (int agent = 0; agent < numAgents; ++agent) {
            Action action = jointAction[agent];
            int agentRow = this.agentRows[agent];
            int agentCol = this.agentCols[agent];
            int boxRow;
            int boxCol;

            switch (action.type) {
                case NoOp:
                    break;
                case Move:
                    destinationRows[agent] = agentRow + action.agentRowDelta;
                    destinationCols[agent] = agentCol + action.agentColDelta;
                    break;
                case Push:
                    // perform Move conflict check
                    destinationRows[agent] = agentRow + action.agentRowDelta;
                    destinationCols[agent] = agentCol + action.agentColDelta;
                    // add box push to agents possible conflicts
                    boxRow = agentRow + action.agentRowDelta;
                    boxCol = agentCol + action.agentColDelta;
                    boxRows[agent] = boxRow + action.boxRowDelta;
                    boxCols[agent] = boxCol + action.boxColDelta;
                    break;
                case Pull:
                    // perform Move conflict check
                    destinationRows[agent] = agentRow + action.agentRowDelta;
                    destinationCols[agent] = agentCol + action.agentColDelta;
                    // add box pull to agents possible conflicts
                    boxRow = agentRow + getOpposite(action.boxRowDelta);
                    boxCol = agentCol + getOpposite(action.boxColDelta);
                    boxRows[agent] = boxRow + action.boxRowDelta;
                    boxCols[agent] = boxCol + action.boxColDelta;
                    break;
            }
        }

        for (int a1 = 0; a1 < numAgents; ++a1) {
            if (jointAction[a1] == Action.NoOp) {
                continue;
            }

            for (int a2 = a1 + 1; a2 < numAgents; ++a2) {
                if (jointAction[a2] == Action.NoOp) {
                    continue;
                }
                // Overlapping box?
                if (boxRows[a1] == boxRows[a2] && boxCols[a1] == boxCols[a2]) {
                    return true;
                }
                // Moving into same cell?
                if (destinationRows[a1] == destinationRows[a2] &&
                        destinationCols[a1] == destinationCols[a2]) {
                    return true;
                }
            }
        }

        return false;
    }

    // check if box is not at the cell
    private boolean noBoxInCell(int row, int col) {
        return this.boxes[row][col] == 0;
    }

    // use trick to convert char to int
    // and normalize it around zero
    private boolean isBoxAndAgentSameColor(char box, Color agentColor) {
        if (box == 0) {
            return false;
        }
        int colorNumber = Character.getNumericValue(box) - 10;
        return boxColors[colorNumber % boxColors.length] == agentColor;
    }

    // compares 0 to move
    // e.g. {0:0} {1:-1} {-1:1}
    private int getOpposite(int move) {
        return Integer.compare(0, move);
    }

    private boolean cellIsFree(int row, int col) {
        return !walls[row][col] && this.noBoxInCell(row, col) && this.agentAt(row, col) == 0;
    }

    private char agentAt(int row, int col) {
        for (int i = 0; i < this.agentRows.length; i++) {
            if (this.agentRows[i] == row && this.agentCols[i] == col) {
                return (char) ('0' + i);
            }
        }
        return 0;
    }

    public Action[][] extractPlan() {
        Action[][] plan = new Action[this.g][];
        State state = this;
        while (state.jointAction != null) {
            plan[state.g - 1] = state.jointAction;
            state = state.parent;
        }
        return plan;
    }

    @Override
    public int hashCode() {
        if (this.hash == 0) {
            final int prime = 31;
            int result = 1;
            result = prime * result + Arrays.hashCode(this.agentColors);
            result = prime * result + Arrays.hashCode(this.boxColors);
            result = prime * result + Arrays.deepHashCode(this.walls);
            result = prime * result + Arrays.deepHashCode(this.goals);
            result = prime * result + Arrays.hashCode(this.agentRows);
            result = prime * result + Arrays.hashCode(this.agentCols);
            for (int row = 0; row < this.boxes.length; ++row) {
                for (int col = 0; col < this.boxes[row].length; ++col) {
                    char c = this.boxes[row][col];
                    if (c != 0) {
                        result = prime * result + (row * this.boxes[row].length + col) * c;
                    }
                }
            }
            this.hash = result;
        }
        return this.hash;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null) {
            return false;
        }
        if (this.getClass() != obj.getClass()) {
            return false;
        }
        State other = (State) obj;
        return Arrays.equals(this.agentRows, other.agentRows) &&
                Arrays.equals(this.agentCols, other.agentCols) &&
                Arrays.equals(this.agentColors, other.agentColors) &&
                Arrays.deepEquals(this.walls, other.walls) &&
                Arrays.deepEquals(this.boxes, other.boxes) &&
                Arrays.equals(this.boxColors, other.boxColors) &&
                Arrays.deepEquals(this.goals, other.goals);
    }

    @Override
    public String toString() {
        StringBuilder s = new StringBuilder();
        for (int row = 0; row < this.walls.length; row++) {
            for (int col = 0; col < this.walls[row].length; col++) {
                if (this.boxes[row][col] > 0) {
                    s.append(this.boxes[row][col]);
                } else if (this.walls[row][col]) {
                    s.append("+");
                } else if (this.agentAt(row, col) != 0) {
                    s.append(this.agentAt(row, col));
                } else {
                    s.append(" ");
                }
            }
            s.append("\n");
        }
        return s.toString();
    }
}
