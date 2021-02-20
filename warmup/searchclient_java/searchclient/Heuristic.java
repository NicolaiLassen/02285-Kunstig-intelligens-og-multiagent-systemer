package searchclient;

import java.util.Comparator;

public abstract class Heuristic
        implements Comparator<State> {

    public Heuristic(State initialState) {
        // Here's a chance to pre-process the static parts of the level.
    }

    public int h(State s) {
        int r = 0;
        for (int row = 0; row < State.goals.length; row++) {
            for (int col = 0; col < State.goals[row].length; col++) {

                char goalTile = State.goals[row][col];
                if (goalTile == 0 || State.walls[row][col]) {
                    continue;
                }

                if (s.boxes[row][col] == goalTile) {
                    r--;
                }

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
