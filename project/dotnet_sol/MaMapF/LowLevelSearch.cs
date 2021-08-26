using System;
using System.Collections.Generic;
using System.Linq;
using MaMapF.Models;
using Priority_Queue;
using Action = MaMapF.Models.Action;

namespace MaMapF
{
    public class LowLevelSearch
    {
        public readonly Level Level;

        public LowLevelSearch(Level level)
        {
            this.Level = level;
        }

        public List<SingleAgentState> GetSingleAgentPlan(char agent, List<Constraint> constraints)
        {
            var goals = Level.Goals[agent];
            var heuristic = new LowLevelHeuristic(goals);


            var frontier = new SimplePriorityQueue<SingleAgentState>();
            var explored = new HashSet<SingleAgentState>();

            var initialState = Level.GetInitialState(agent);
            frontier.Enqueue(initialState, 0);

            while (frontier.Count != 0)
            {
                Console.Error.WriteLine(frontier.Count);
                var state = frontier.Dequeue();
                explored.Add(state);

                // Console.Error.WriteLine(frontier.Count);
                // Console.Error.WriteLine(state);
                // if (state.G > 10) Environment.Exit(0);

                if (Level.IsAgentGoalState(state))
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
                        s.H = heuristic.GetHeuristic(s);
                        
                        // greedy
                        frontier.Enqueue(s, s.H);

                        // astar
                        // frontier.Enqueue(s, s.F);
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

            // if (constraints.Any())
            // {
            //     Console.Error.WriteLine("QQQQQQQQQQQQQQQQQQQQQQQQQQQ");
            //     constraints.ForEach(c => Console.Error.WriteLine(c));
            //
            //     Console.Error.WriteLine("STATE");
            //     Console.Error.WriteLine(state);
            //
            //     Console.Error.WriteLine("EXPANDED");
            //     states.ForEach(s => Console.Error.WriteLine(s));
            //     Environment.Exit(0);
            // }


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

            // Console.Error.WriteLine(nextState);

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