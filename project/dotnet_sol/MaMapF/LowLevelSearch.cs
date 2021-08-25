using System.Collections.Generic;
using MaMapF.Models;
using Priority_Queue;

namespace MaMapF
{
    public class LowLevelSearch
    {
        public static List<SingleAgentState> GetSingleAgentPlan(SingleAgentState initialState)
        {
            var frontier = new SimplePriorityQueue<SingleAgentState>();
            var explored = new HashSet<SingleAgentState>();

            frontier.Enqueue(initialState, 0);

            while (frontier.Count != 0)
            {
                var state = frontier.Dequeue();
                explored.Add(state);

                if (IsGoalState(state))
                {
                    return GetSingleAgentSolutionFromState(state);
                }

                var expandedStates = ExpandSingleAgentState(state);
                foreach (var s in expandedStates)
                {
                    if (!explored.Contains(s) && !frontier.Contains(s))
                    {
                        frontier.Enqueue(s, s.F);
                    }
                }
            }


            return null;
        }


        public static bool IsGoalState(SingleAgentState state)
        {
            return false;
        }

        public static List<SingleAgentState> GetSingleAgentSolutionFromState(SingleAgentState goal)
        {
            var solution = new List<SingleAgentState>();
            var state = goal;

            while (state.Action != null)
            {
                solution.Insert(0, state);
                state = state.Parent;
            }

            solution.Insert(0, state);
            return solution;
        }

        public static List<SingleAgentState> ExpandSingleAgentState(SingleAgentState state)
        {
            return new List<SingleAgentState>();
        }
    }
}


// def get_low_level_plan(initial_state: State, constraints=[]):
// frontier = FrontierBestFirst()
// explored = set()
//
// frontier.add(initial_state)
// while True:
// if frontier.is_empty():
// break
//
// state = frontier.pop()
//
// if state.is_goal_state():
// return state.get_solution()
//
// explored.add(state)
//
// for state in state.expand_state(constraints):
// is_not_frontier = not frontier.contains(state)
// is_explored = state not in explored
// if is_not_frontier and is_explored:
// frontier.add(state)