using System;
using System.Collections.Generic;
using System.Linq;
using MaMapF.Models;
using Priority_Queue;
using Action = MaMapF.Models.Action;

namespace MaMapF.Handlers
{
    public enum SearchType
    {
        BFS,
        GREEDY,
        ASTAR
    }

    public class SingleAgentSearchHandler
    {
        public static readonly SearchType SearchType = SearchType.ASTAR;
        public static readonly bool PrintProgress = false; // Check map progress, warning: very slow 

        public static List<SingleAgentState> Search(
            SingleAgentProblem problem,
            List<Constraint> constraints
        )
        {
            var initialState = problem.InitialState;
            var goals = problem.Goals;
            
            if (PrintProgress)
            {
                Console.Error.WriteLine(problem);
            }

            // Skip node has unreachable constraints
            var maxDistanceConstraint = 0;
            var maxConstraintStep = 0;
            foreach (var constraint in constraints)
            {
                var distanceToConstraint = Position.Distance(initialState.Agent.Position, constraint.Position);
                if (distanceToConstraint > constraint.Step + 1)
                {
                    return null;
                }

                if (maxDistanceConstraint >= distanceToConstraint) continue;

                maxDistanceConstraint = distanceToConstraint;
                maxConstraintStep = constraint.Step;
            }

            var heuristic = new SingleAgentHeuristic(goals, constraints);

            var frontier = new SimplePriorityQueue<SingleAgentState>();
            var explored = new HashSet<SingleAgentState>();

            frontier.Enqueue(initialState, 0);

            while (frontier.Count != 0)
            {
                var state = frontier.Dequeue();
                explored.Add(state);

                if (problem.Type == SingleAgentProblemType.NULL)
                {
                    if (maxConstraintStep > state.G + maxDistanceConstraint)
                    {
                        var nextNoOpState = CreateNextState(state, Action.NoOp);
                        frontier.Enqueue(nextNoOpState, maxConstraintStep - state.G);
                        continue;
                    }
                }

                if (PrintProgress)
                {
                    Console.Error.WriteLine(state);
                }

                if (IsGoalState(state, goals, constraints))
                {
                    return GetSingleAgentSolutionFromState(state);
                }

                var expandedStates = ExpandSingleAgentState(state, constraints);
                foreach (var s in expandedStates)
                {
                    // skip if state is already in list of frontiers
                    if (frontier.Contains(s)) continue;

                    // skip if state is already explored
                    if (explored.Contains(s)) continue;

                    s.H = heuristic.GetHeuristic(problem, s);
                    var priority = GetPriority(s);
                    frontier.Enqueue(s, priority);
                }
            }

            return null;
        }

        private static int GetPriority(SingleAgentState s)
        {
            if (SearchType == SearchType.ASTAR) return s.F;
            if (SearchType == SearchType.GREEDY) return s.H;
            return s.G;
        }

        private static bool IsGoalState(SingleAgentState state, List<MapItem> goals, List<Constraint> constraints)
        {
            // false if there is a future constraint
            var futureConstraint = constraints.FirstOrDefault(c => c.Step > state.G);
            if (futureConstraint != null)
            {
                return false;
            }

            // false if satisfied goals != goals
            var counter = goals.Count(goal => state.AllMapItems.Any(item => item.Equals(goal)));
            return counter == goals.Count;
        }


        private static List<SingleAgentState> ExpandSingleAgentState(SingleAgentState state,
            List<Constraint> constraints)
        {
            var states = new List<SingleAgentState>();
            foreach (var action in Action.AllActions)
            {
                if (IsValidAction(state, action))
                {
                    var lastPastConstraint = constraints.LastOrDefault(c => c.Step <= state.G);
                    var nextState = CreateNextState(state, action);

                    nextState.PastConstraint = lastPastConstraint;

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
            nextState.ROWS = state.ROWS;
            nextState.COLS = state.COLS;

            // Use reference to walls
            nextState.Walls = state.Walls;

            // OH MY FUCKING GOD, copy all reference
            nextState.Agent = new MapItem(state.Agent.UID, state.Agent.Value, state.Agent.Position);
            nextState.Boxes = state.Boxes.Select(b => b).ToList();
            nextState.BoxWalls = state.BoxWalls.Select(b => b).ToList();

            // if (action.Type == ActionType.NoOp)
            if (action.Type == ActionType.Move)
            {
                nextState.Agent.Position = state.Agent.Position.Next(action.AgentRowDelta, action.AgentColumnDelta);
            }

            if (action.Type == ActionType.Push)
            {
                nextState.Agent.Position = state.Agent.Position.Next(action.AgentRowDelta, action.AgentColumnDelta);
                var nextBoxPosition = nextState.Agent.Position.Next(action.BoxRowDelta, action.BoxColumnDelta);
                nextState.Boxes = state.Boxes.Select(b => b.Position.Equals(nextState.Agent.Position)
                    ? new MapItem(b.UID, b.Value, nextBoxPosition)
                    : b).ToList();
            }

            if (action.Type == ActionType.Pull)
            {
                var boxPosition = state.Agent.Position.Next(action.BoxRowDelta * -1, action.BoxColumnDelta * -1);
                var nextBoxPosition = state.Agent.Position.Next(0, 0);
                nextState.Agent.Position = state.Agent.Position.Next(action.AgentRowDelta, action.AgentColumnDelta);
                nextState.Boxes = state.Boxes.Select(b => b.Position.Equals(boxPosition)
                    ? new MapItem(b.UID, b.Value, nextBoxPosition)
                    : b).ToList();
            }

            nextState.G = state.G + 1;
            // CALC LATER hmmmm
            nextState.H = 0;

            return nextState;
        }

        private static bool BreaksConstraint(SingleAgentState state, List<Constraint> constraints)
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

        private static bool IsValidAction(SingleAgentState state, Action action)
        {
            if (action.Type == ActionType.NoOp)
            {
                return true;
            }

            if (action.Type == ActionType.Move)
            {
                var nextAgentPosition = state.Agent.Position.Next(action.AgentRowDelta, action.AgentColumnDelta);
                return state.IsFree(nextAgentPosition);
            }

            if (action.Type == ActionType.Push)
            {
                // check that next agent position is box and next box position is free
                var nextAgentPosition = state.Agent.Position.Next(action.AgentRowDelta, action.AgentColumnDelta);
                var nextBoxPosition = nextAgentPosition.Next(action.BoxRowDelta, action.BoxColumnDelta);
                return state.IsBox(nextAgentPosition) && state.IsFree(nextBoxPosition);
            }

            if (action.Type == ActionType.Pull)
            {
                // check that next agent position is free and box position is box
                var nextAgentPosition = state.Agent.Position.Next(action.AgentRowDelta, action.AgentColumnDelta);
                var boxPosition = state.Agent.Position.Next(action.BoxRowDelta * -1, action.BoxColumnDelta * -1);
                return state.IsFree(nextAgentPosition) && state.IsBox(boxPosition);
            }

            return false;
        }

        private static List<SingleAgentState> GetSingleAgentSolutionFromState(SingleAgentState goal)
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