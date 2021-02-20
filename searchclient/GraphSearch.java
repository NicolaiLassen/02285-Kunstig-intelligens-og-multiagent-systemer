package searchclient;

import java.util.HashSet;

public class GraphSearch {

    public static Action[][] search(State initialState, Frontier frontier) {
        boolean outputFixedSolution = false;

        if (outputFixedSolution) {
            //Part 1:
            //The agents will perform the sequence of actions returned by this method.
            //Try to solve a few levels by hand, enter the found solutions below, and run them:

            return new Action[][]{
                    {Action.MoveE},
                    {Action.MoveE},
                    {Action.MoveE},
                    {Action.MoveE},
                    {Action.MoveE},
                    {Action.PushEE}
            };
        } else {
            int iterations = 0;

            frontier.add(initialState);
            HashSet<State> explored = new HashSet<>();

            while (true) {
                // print a status message every 10000 iteration
                if (++iterations % 1000 == 0) {
                    printSearchStatus(explored, frontier);
                }

                // if the frontier is empty then return failure
                if (frontier.isEmpty()) {
                    return null;
                }
                // choose a leaf node and remove it from the frontier
                State node = frontier.pop();
                // if the node contains a goal state then return the corresponding solution
                if (node.isGoalState()) {
                    printSearchStatus(explored, frontier);
                    return node.extractPlan();
                }
                // add the node to the explored set
                explored.add(node);
                // expand the chosen node, adding the resulting nodes to the frontier
                node.getExpandedStates().forEach(state -> {
                    // only if not in the frontier or explored set
                    // System.err.println(state.toString());
                    if (!frontier.contains(state) && !explored.contains(state)) {
                        frontier.add(state);
                    }
                });
            }
        }
    }

    private static long startTime = System.nanoTime();

    private static void printSearchStatus(HashSet<State> explored, Frontier frontier) {
        String statusTemplate = "#Expanded: %,8d, #Frontier: %,8d, #Generated: %,8d, Time: %3.3f s\n%s\n";
        double elapsedTime = (System.nanoTime() - startTime) / 1_000_000_000d;
        System.err.format(statusTemplate, explored.size(), frontier.size(), explored.size() + frontier.size(),
                elapsedTime, Memory.stringRep());
    }
}
