﻿using System;
using System.Collections.Generic;
using System.Linq;
using MaMapF.Models;
using Priority_Queue;
using Action = MaMapF.Models.Action;

namespace MaMapF
{
    public enum SearchType
    {
        BFS,
        GREEDY,
        ASTAR
    }

    public class LowLevelSearch
    {
        public static readonly SearchType SearchType = SearchType.GREEDY;
        public static readonly int MaxActionRepeat = -1;

        public static List<SingleAgentState> GetSingleAgentPlan(
            SingleAgentState initialState,
            List<MapItem> goals,
            List<Constraint> constraints
        )
        {
            var heuristic = new LowLevelHeuristic(goals, constraints);

            var frontier = new SimplePriorityQueue<SingleAgentState>();
            var explored = new HashSet<SingleAgentState>();

            frontier.Enqueue(initialState, 0);

            while (frontier.Count != 0)
            {
                var state = frontier.Dequeue();
                explored.Add(state);

                // Console.Error.WriteLine(frontier.Count);
                // if (state.G > 10) Environment.Exit(0);
                Console.Error.WriteLine(state);

                if (IsGoalState(state, goals))
                {
                    return GetSingleAgentSolutionFromState(state);
                }

                var expandedStates = ExpandSingleAgentState(state, constraints);
                foreach (var s in expandedStates)
                {
                    var isNotFrontier = !frontier.Contains(s);
                    var isNotExplored = !explored.Contains(s);
                    var isNotExploredMax = IsNotExploredMaxNoOp(explored, s);
                    if (isNotFrontier && isNotExplored && isNotExploredMax)
                    {
                        s.H = heuristic.GetHeuristic(s);
                        var priority = GetPriority(s);
                        frontier.Enqueue(s, priority);
                    }
                }
            }

            return null;
        }

        private static bool IsNotExploredMaxNoOp(HashSet<SingleAgentState> explored, SingleAgentState state)
        {
            if (MaxActionRepeat == -1) return true;
            var s = CreateNextState(state, Action.NoOp);
            s.G -= MaxActionRepeat;
            var isNotExplored = !explored.Contains(s);
            return isNotExplored;
        }

        private static int GetPriority(SingleAgentState s)
        {
            if (SearchType == SearchType.GREEDY) return s.H;
            if (SearchType == SearchType.ASTAR) return s.F;
            return s.G;
        }


        private static bool IsGoalState(SingleAgentState state, List<MapItem> goals)
        {
            var counter = goals.Count(agentGoal =>
                state.Map[agentGoal.Position.Row][agentGoal.Position.Column] == agentGoal.Value);
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
                    var nextState = CreateNextState(state, action);
                    if (!BreaksConstraint(nextState, constraints))
                    {
                        nextState.Counter = constraints.Count;
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
            nextState.Boxes = state.Boxes.Select(b => b).ToList();
            nextState.Map = state.Map.Select(item => item.Select(e => e).ToList()).ToList();


            // if (action.Type == ActionType.NoOp)
            if (action.Type == ActionType.Move)
            {
                nextState.AgentPosition = state.AgentPosition.Next(action.AgentRowDelta, action.AgentColumnDelta);

                // Update map
                nextState.Map[nextState.AgentPosition.Row][nextState.AgentPosition.Column] = state.Agent;
                nextState.Map[state.AgentPosition.Row][state.AgentPosition.Column] = ' ';
            }

            if (action.Type == ActionType.Push)
            {
                nextState.AgentPosition = state.AgentPosition.Next(action.AgentRowDelta, action.AgentColumnDelta);
                var nextBoxPosition = nextState.AgentPosition.Next(action.BoxRowDelta, action.BoxColumnDelta);
                nextState.Boxes = state.Boxes.Select(b =>
                    b.Position.Equals(nextState.AgentPosition) ? new MapItem(b.Value, nextBoxPosition) : b).ToList();


                // update map
                var boxValue = state.Map[nextState.AgentPosition.Row][nextState.AgentPosition.Column];
                nextState.Map[nextBoxPosition.Row][nextBoxPosition.Column] = boxValue;
                nextState.Map[nextState.AgentPosition.Row][nextState.AgentPosition.Column] = state.Agent;
                nextState.Map[state.AgentPosition.Row][state.AgentPosition.Column] = ' ';
            }

            if (action.Type == ActionType.Pull)
            {
                var boxPosition = state.AgentPosition.Next(action.BoxRowDelta * -1, action.BoxColumnDelta * -1);
                var nextBoxPosition = state.AgentPosition.Next(0, 0);
                nextState.AgentPosition = state.AgentPosition.Next(action.AgentRowDelta, action.AgentColumnDelta);
                nextState.Boxes = state.Boxes
                    .Select(b => b.Position.Equals(boxPosition) ? new MapItem(b.Value, nextBoxPosition) : b).ToList();

                // update map
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