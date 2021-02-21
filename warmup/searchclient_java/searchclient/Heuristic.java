package searchclient;

import java.util.ArrayList;
import java.util.Comparator;

class Item {
    public char id;
    public int row;
    public int col;
    public int distanceToEnd;

    Item(char id, int row, int col) {
        this.id = id;
        this.row = row;
        this.col = col;
        this.distanceToEnd = 0;
    }
}

public abstract class Heuristic implements Comparator<State> {

    private final ArrayList<Item> goals = new ArrayList<>();

    public Heuristic(State initialState) {
        // init informed search graph for every client
        var rowCount = initialState.getRowCount();
        var colCount = initialState.getColCount();

        for (int row = 0; row < rowCount; row++) {
            for (int col = 0; col < colCount; col++) {
                var GoalItem = State.goals[row][col];
                if (GoalItem == 0) {
                    continue;
                }
                var goalItem = new Item(GoalItem, row, col);
                goals.add(goalItem);
            }
        }
    }

    // use minimum computation to find level path
    public int h(State s) {
        return hDistance(s);
    }

    public int hDistance(State s) {
        var rowCount = s.getRowCount();
        var colCount = s.getColCount();
        ArrayList<Item> targetsBoxes = new ArrayList<>(65536);

        for (int row = 0; row < rowCount; row++) {
            for (int col = 0; col < colCount; col++) {

                var tileId = s.boxes[row][col];
                if (tileId == 0 || State.walls[row][col]) {
                    continue;
                }

                // boxes
                int distanceMin = 1000;
                Item selectedTargetItem = null;
                for (Item goal : goals) {
                    if (goal.id == tileId) {
                        var distance = calcManhattan(goal.row, row, goal.col, col);
                        if (distance < distanceMin) {
                            selectedTargetItem = new Item(tileId, row, col);
                            selectedTargetItem.distanceToEnd = distance;
                            distanceMin = distance;
                        }
                    }
                }
                if (selectedTargetItem != null) {
                    targetsBoxes.add(selectedTargetItem);
                }
            }
        }

        int r = 0;
        for (Item targetsBox : targetsBoxes) {
            r += targetsBox.distanceToEnd;
        }

        // check same color
        // TODO: This should messure the distance from
        // agent to box
        // if box is at goal pos the don't do this
        for (int i = 0; i < s.agentCols.length; i++) {
            // Should count the box that is close as a more valid target
            // if the box is has a distance of 0 from goal then
            // don't count the box
            int distanceMin = 100;
            for (Item targetsBox : targetsBoxes) {
                var distance = calcManhattan(targetsBox.row, s.agentRows[i], targetsBox.col, s.agentRows[i]);
                if (distance < distanceMin) {
                    distanceMin = distance;
                }
            }
            r += distanceMin;
        }

        return r;
    }

    private char getIntAsCharKey(int k) {
        return Character.forDigit(k, 10);
    }

    // diagonal moves ar not possible
    // therefore use the manhattan distance
    private int calcManhattan(int p1r, int p2r, int p1c, int p2c) {
        return Math.abs(p1r - p2r) + Math.abs(p1c - p2c);
    }

    public int hGoalCount(State s) {
        // goal count
        int r = 0;
        // loop over the map
        for (int row = 0; row < State.goals.length; row++) {
            for (int col = 0; col < State.goals[row].length; col++) {
                // check if tile is a wall or a empty tile
                char goalTile = State.goals[row][col];
                if (goalTile == 0 || State.walls[row][col]) {
                    continue;
                }
                // is box and in goal position
                if (s.boxes[row][col] == goalTile) {
                    r--;
                }
                // is agent and in goal position
                for (int k = 0; k < s.agentRows.length; k++) {
                    if (s.agentRows[k] == row && s.agentCols[k] == col) {
                        if (k == Character.getNumericValue(goalTile)) {
                            r--;
                        }
                    }
                }
            }
        }
        return r;
    }

    public abstract int f(State s);

    @Override
    public int compare(State s1, State s2) {
        return this.f(s1) - this.f(s2);
    }
}

class HeuristicAStar
        extends Heuristic {
    public HeuristicAStar(State initialState) {
        super(initialState);
    }

    @Override
    public int f(State s) {
        return s.g() + this.h(s);
    }

    @Override
    public String toString() {
        return "A* evaluation";
    }
}

class HeuristicWeightedAStar
        extends Heuristic {
    private int w;

    public HeuristicWeightedAStar(State initialState, int w) {
        super(initialState);
        this.w = w;
    }

    @Override
    public int f(State s) {
        return s.g() + this.w * this.h(s);
    }

    @Override
    public String toString() {
        return String.format("WA*(%d) evaluation", this.w);
    }
}

class HeuristicGreedy
        extends Heuristic {
    public HeuristicGreedy(State initialState) {
        super(initialState);
    }

    @Override
    public int f(State s) {
        return this.h(s);
    }

    @Override
    public String toString() {
        return "greedy evaluation";
    }
}
