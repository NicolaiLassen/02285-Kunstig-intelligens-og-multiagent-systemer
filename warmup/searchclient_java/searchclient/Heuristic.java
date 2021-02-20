package searchclient;

import java.util.ArrayDeque;
import java.util.Comparator;
import java.util.HashMap;

public abstract class Heuristic
        implements Comparator<State> {

    private final HashMap<Character, int[]> GoalState = new HashMap<>(65536);
    private final HashMap<Character, int[]> ObservableElements = new HashMap<>(65536);
    private final ArrayDeque<Character> Elements = new ArrayDeque<Character>(65536);

    public Heuristic(State initialState) {

        for (int row = 0; row < State.goals.length; row++) {
            for (int col = 0; col < State.goals[row].length; col++) {

            }
        }

    }

    private int CalcManhattan(int p1r, int p1c, int p2r, int p2c) {
        return Math.abs(p1r - p2r) + Math.abs(p1c - p2c);
    }


    public int h(State s) {
        int r = 0;
        for (int i = 0; i < Elements.size(); i++) {

        }
        return r;
    }

    // GOAL COUNT
//    public int h(State s) {
//        int r = 0;
//        for (int row = 0; row < State.goals.length; row++) {
//            for (int col = 0; col < State.goals[row].length; col++) {
//
//                char goalTile = State.goals[row][col];
//                if (goalTile == 0 || State.walls[row][col]) {
//                    continue;
//                }
//
//                if (s.boxes[row][col] == goalTile) {
//                    r--;
//                }
//
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
