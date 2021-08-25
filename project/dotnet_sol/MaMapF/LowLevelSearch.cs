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


        public static List<SingleAgentState> ExpandSingleAgentState(SingleAgentState state)
        {
            var states = new List<SingleAgentState>();
            foreach (var action in Action.AllActions)
            {
                if ()
            }
        }

        public static bool IsActionValid(SingleAgentState state, Action action)
        {
            if (action.Type == ActionType.NoOp)
            {
                return true;
            }

            if (action.Type == ActionType.Move)
            {
                var nextAgentPosition = state.AgentPosition.Next(action.AgentRowDelta, action.AgentColumnDelta);
                return state.IsFree(nextAgentPosition);
            }
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
    }
}