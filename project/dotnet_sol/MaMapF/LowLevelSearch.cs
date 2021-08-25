using System.Collections.Generic;
using System.Linq;
using MaMapF.Models;
using Priority_Queue;

namespace MaMapF
{
    public class LowLevelSearch
    {
        public static List<SingleAgentState> GetSingleAgentPlan(
            Level level,
            SingleAgentState initialState,
            List<Constraint> constraints
        )
        {
            var frontier = new SimplePriorityQueue<SingleAgentState>();
            var explored = new HashSet<SingleAgentState>();

            frontier.Enqueue(initialState, 0);

            while (frontier.Count != 0)
            {
                var state = frontier.Dequeue();
                explored.Add(state);

                // Console.Error.WriteLine(state);

                if (level.IsAgentGoalState(state))
                {
                    return GetSingleAgentSolutionFromState(state);
                }

                var expandedStates = ExpandSingleAgentState(state, constraints);
                foreach (var s in expandedStates)
                {
                    var isNotFrontier = !frontier.Contains(s);
                    var isNotExplored = !explored.Contains(s);
                    if (isNotFrontier && isNotExplored)
                    {
                        var f = level.GetHeuristic(s);
                        frontier.Enqueue(s, f);
                    }
                }
            }


            return null;
        }

        public static List<SingleAgentState> ExpandSingleAgentState(SingleAgentState state,
            List<Constraint> constraints)
        {
            var states = new List<SingleAgentState>();
            foreach (var action in Action.AllActions)
            {
                if (IsValidAction(state, action))
                {
                    var nextState = CreateNextState(state, action);
                    if (!BreaksConstraint(nextState, constraints))
                    {
                        states.Add(nextState);
                    }
                }
            }

            return states;
        }

        public static SingleAgentState CreateNextState(SingleAgentState state, Action action)
        {
            var nextState = new SingleAgentState();
            nextState.Parent = state;
            nextState.Action = action;

            nextState.Agent = state.Agent;
            nextState.AgentPosition = state.AgentPosition;
            nextState.Map = state.Map.Select(item => item.Select(e => e).ToList()).ToList();


            // if (action.Type == ActionType.NoOp)
            if (action.Type == ActionType.Move)
            {
                nextState.AgentPosition = state.AgentPosition.Next(action.AgentRowDelta, action.AgentColumnDelta);
                nextState.Map[nextState.AgentPosition.Row][nextState.AgentPosition.Column] = state.Agent;
                nextState.Map[state.AgentPosition.Row][state.AgentPosition.Column] = ' ';
            }

            if (action.Type == ActionType.Push)
            {
                nextState.AgentPosition = state.AgentPosition.Next(action.AgentRowDelta, action.AgentColumnDelta);
                var boxValue = state.Map[nextState.AgentPosition.Row][nextState.AgentPosition.Column];
                var nextBoxPosition = nextState.AgentPosition.Next(action.BoxRowDelta, action.BoxColumnDelta);

                nextState.Map[nextBoxPosition.Row][nextBoxPosition.Column] = boxValue;
                nextState.Map[nextState.AgentPosition.Row][nextState.AgentPosition.Column] = state.Agent;
                nextState.Map[state.AgentPosition.Row][state.AgentPosition.Column] = ' ';
            }

            if (action.Type == ActionType.Pull)
            {
                nextState.AgentPosition = state.AgentPosition.Next(action.AgentRowDelta, action.AgentColumnDelta);
                var boxPosition = state.AgentPosition.Next(action.BoxRowDelta * -1, action.BoxColumnDelta * -1);
                var boxValue = state.Map[boxPosition.Row][boxPosition.Column];

                nextState.Map[nextState.AgentPosition.Row][nextState.AgentPosition.Column] = state.Agent;
                nextState.Map[state.AgentPosition.Row][state.AgentPosition.Column] = boxValue;
                nextState.Map[boxPosition.Row][boxPosition.Column] = ' ';
            }

            nextState.G = state.G + 1;

            // CALC LATER hmmmm
            nextState.H = 0;

            return nextState;
        }

        public static bool BreaksConstraint(SingleAgentState state, List<Constraint> constraints)
        {
            foreach (var constraint in constraints.Where(c => c.Step == state.G))
            {
                if (!state.IsFree(constraint.Position))
                {
                    return true;
                }
            }

            return false;
        }

        public static bool IsValidAction(SingleAgentState state, Action action)
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

            if (action.Type == ActionType.Push)
            {
                // check that next agent position is box and next box position is free
                var nextAgentPosition = state.AgentPosition.Next(action.AgentRowDelta, action.AgentColumnDelta);
                var nextBoxPosition = nextAgentPosition.Next(action.BoxRowDelta, action.BoxColumnDelta);
                return state.IsBox(nextAgentPosition) && state.IsFree(nextBoxPosition);
            }

            if (action.Type == ActionType.Pull)
            {
                // check that next agent position is free and box position is box
                var nextAgentPosition = state.AgentPosition.Next(action.AgentRowDelta, action.AgentColumnDelta);
                var boxPosition = state.AgentPosition.Next(action.BoxRowDelta * -1, action.BoxColumnDelta * -1);
                return state.IsFree(nextAgentPosition) && state.IsBox(boxPosition);
            }

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
    }
}