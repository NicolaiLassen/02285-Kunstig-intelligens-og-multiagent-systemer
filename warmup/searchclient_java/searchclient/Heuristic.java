package searchclient;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;

public abstract class Heuristic
        implements Comparator<State> {

    // keep all goal state col and row
    // int[] where [0] = row and [1] = col
    private final HashMap<Character, HashMap<int[], Integer>> informedGraph = new HashMap<>();
    private final HashMap<Character, ArrayList<int[]>> goalRowCol = new HashMap<>(65536);

    public Heuristic(State initialState) {
        // init informed search graph for every client
        // informedGraph = new int[initialState.agentCols.length][State.goals.length][State.goals[0].length];

        // Set all goal positions
        for (int row = 0; row < State.goals.length; row++) {
            for (int col = 0; col < State.goals[row].length; col++) {
                // discard non goal object
                char goalTile = State.goals[row][col];
                if (goalTile == 0 || State.walls[row][col]) {
                    return;
                }
                // does goalRowCol contain the char
                if (!goalRowCol.containsKey(goalTile)) {
                    // init new map of goal pos
                    goalRowCol.put(State.goals[row][col], new ArrayList<>());
                }
                // add goal pos
                goalRowCol.get(goalTile).add(new int[]{row, col});
            }
        }

        // create informed graph for each agent
        for (int k = 0; k < initialState.agentRows.length; k++) {
            // fill in informed graph
            for (int row = 0; row < State.goals.length; row++) {
                for (int col = 0; col < State.goals[row].length; col++) {

                    // push state with walls to back'
                    // these can't be reached
                    if (State.walls[row][col]) {
                        informedGraph[k][row][col] = 10000;
                    }
                }
            }
        }
    }

    // use minimum computation to find level path
    public int h(State s) {
        int r = 0;
        // For each of the clients find the min value of the level
        // using the informed Graph
        for (int k = 0; k < s.agentRows.length; k++) {
            r += informedGraph[k][s.agentRows[k]][s.agentCols[k]];
        }
        return r;
    }

    // diagonal moves ar not possible
    // therefore use the manhattan distance
    private int CalcManhattan(int p1r, int p1c, int p2r, int p2c) {
        return Math.abs(p1r - p2r) + Math.abs(p1c - p2c);
    }

//    public int h(State s) {
//        // goal count
//        int r = 0;
//        // loop over the map
//        for (int row = 0; row < State.goals.length; row++) {
//            for (int col = 0; col < State.goals[row].length; col++) {
//                // check if tile is a wall or a empty tile
//                char goalTile = State.goals[row][col];
//                if (goalTile == 0 || State.walls[row][col]) {
//                    continue;
//                }
//                // is box and in goal position
//                if (s.boxes[row][col] == goalTile) {
//                    r--;
//                }
//                // is agent and in goal position
//                for (int k = 0; k < s.agentRows.length; k++) {
//                    if (s.agentRows[k] == row && s.agentCols[k] == col) {
//                        if (k == Character.getNumericValue(goalTile)) {
//                            r--;
//                        }
//                    }
//                }
//            }
//        }
//        return r;
//    }

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
