package searchclient;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;


class Item {
    public char id;
    public int row;
    public int col;

    Item(char id, int row, int col) {
        this.id = id;
        this.row = row;
        this.col = col;
    }
}


public abstract class Heuristic implements Comparator<State> {


    private ArrayList<Item> allGoals = new ArrayList<>();
    private HashMap<Character, int[][]> informedMap = new HashMap<>();


    public Heuristic(State initialState) {
        // init informed search graph for every client
        var rowCount = initialState.getRowCount();
        var colCount = initialState.getColCount();


        for (int row = 0; row < rowCount; row++) {
            for (int col = 0; col < colCount; col++) {
                var id = State.goals[row][col];
                if (State.walls[row][col] || id == 0) {
                    continue;
                }
                var item = new Item(id, row, col);
                allGoals.add(item);
            }
        }

        for (Item item : allGoals) {
            informedMap.put(item.id, new int[rowCount][colCount]);
        }


        for (int row = 0; row < rowCount; row++) {
            for (int col = 0; col < colCount; col++) {
                if (State.walls[row][col]) {
                    continue;
                }

                for (Item item : allGoals) {
                    var rowDiff = Math.abs(item.row - row);
                    var colDiff = Math.abs(item.col - col);
                    var diff = rowDiff + colDiff;
                    informedMap.get(item.id)[row][col] = diff;
                }
            }
        }


        for (Item item : allGoals) {
            System.err.println("");
            System.err.println("ITEM: " + item.id);

            for (int row = 0; row < rowCount; row++) {
                for (int col = 0; col < colCount; col++) {
                    var val = informedMap.get(item.id)[row][col];
                    System.err.print("" + val + "\t");
                }
                System.err.println("");
            }
        }

    }

    // use minimum computation to find level path
    public int h(State s) {
        return hDistance(s);
    }

    public int hDistance(State s) {
        int r = 0;
        var rowCount = s.getRowCount();
        var colCount = s.getColCount();
        for (int row = 0; row < rowCount; row++) {
            for (int col = 0; col < colCount; col++) {
                if (State.walls[row][col]) {
                    continue;
                }

                for (Item item : allGoals) {
                    r += informedMap.get(item.id)[row][col];
                }
            }
        }

        return r;
    }

    // diagonal moves ar not possible
    // therefore use the manhattan distance
    private int CalcManhattan(int p1r, int p1c, int p2r, int p2c) {
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
