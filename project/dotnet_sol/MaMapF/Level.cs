using System;
using System.Collections.Generic;
using System.Linq;
using MaMapF.Models;

namespace MaMapF
{
    public class Level
    {
        public Dictionary<char, string> Colors { get; set; }
        public List<char> Agents { get; set; }

        public List<List<char>> InitialMap { get; set; }
        public List<List<char>> GoalMap { get; set; }
        public Dictionary<char, List<Goal>> Goals { get; set; }

        public Dictionary<char, SingleAgentState> AgentInitialStates { get; set; } =
            new Dictionary<char, SingleAgentState>();

        public Level(
            Dictionary<char, string> colors,
            List<char> agents,
            List<List<char>> initialMap,
            List<List<char>> goalMap,
            Dictionary<char, List<Goal>> goals
        )
        {
            Colors = colors;
            Agents = agents;
            InitialMap = initialMap;
            GoalMap = goalMap;
            Goals = goals;
            AgentInitialStates = new Dictionary<char, SingleAgentState>();

            foreach (var agent in agents)
            {
                AgentInitialStates.Add(agent, CreateAgentInitialState(agent));
            }
        }

        private SingleAgentState CreateAgentInitialState(char agent)
        {
            var map = new List<List<char>>(InitialMap);
            var agentColor = Colors[agent];
            var agentPosition = new Position();

            for (var row = 0; row < InitialMap.Count; row++)
            {
                for (var col = 0; col < InitialMap[row].Count; col++)
                {
                    var c = InitialMap[row][col];
                    if (c == '+' || c == ' ')
                    {
                        continue;
                    }

                    if (c == agent)
                    {
                        agentPosition = new Position(row, col);
                        continue;
                    }

                    if (char.IsDigit(c))
                    {
                        map[row][col] = ' ';
                    }

                    if (!char.IsLetter(c)) continue;

                    if (Colors[c] != agentColor)
                    {
                        map[row][col] = ' ';
                    }
                }
            }

            return new SingleAgentState
            {
                G = 0,
                Map = map,
                Agent = agent,
                AgentPosition = agentPosition
            };
        }

        public SingleAgentState GetInitialState(char agent) => AgentInitialStates[agent];

        public bool IsAgentGoalState(SingleAgentState state)
        {
            var agentGoals = Goals[state.Agent];
            var counter = agentGoals.Count(agentGoal => state.Map[agentGoal.Row][agentGoal.Column] == agentGoal.Item);
            return counter == Goals[state.Agent].Count;
        }

        public int GetHeuristic(SingleAgentState state)
        {
            var h = 0;
            var boxGoals = Goals[state.Agent].Where(goal => goal.Item != state.Agent);


            // Find box closest to goal position not already taken
            var minBoxPosition = new Position();
            var minBoxDistance = Int32.MaxValue;
            for (var row = 0; row < state.Map.Count; row++)
            {
                for (var col = 0; col < state.Map[row].Count; col++)
                {
                    var c = state.Map[row][col];
                    foreach (var boxGoal in boxGoals)
                    {
                        // skip taken goals
                        if (state.Map[boxGoal.Row][boxGoal.Column] == boxGoal.Item)
                        {
                            continue;
                        }

                        if (c == boxGoal.Item)
                        {
                            var dist = Math.Abs(row - boxGoal.Row) + Math.Abs(col - boxGoal.Column);
                            if (dist < minBoxDistance)
                            {
                                minBoxPosition = new Position(row, col);
                                minBoxDistance = dist;
                            }
                        }
                    }
                }
            }

            // Add distance from closest box to non-taken goal
            h += minBoxDistance;

            // Add distance from agent to minBox
            var agentDistanceToMinBox = Math.Abs(state.AgentPosition.Row - minBoxPosition.Row) +
                                        Math.Abs(state.AgentPosition.Column - minBoxPosition.Column);
            h += agentDistanceToMinBox;


            // Add manhatten distance to agent goal
            if (minBoxDistance == Int32.MaxValue)
            {

                var agentGoal = Goals[state.Agent].FirstOrDefault(goal => goal.Item == state.Agent);
                if (agentGoal != null)
                {
                    var dist = Math.Abs(agentGoal.Row - state.AgentPosition.Row) +
                               Math.Abs(agentGoal.Column - state.AgentPosition.Column);
                    h += dist;
                }

            }
            

            return h;

            // var h = GetHeuristicGoalCount(state);
            // var h = GetHeuristicMaxManhattenDist(state);
            // return h + state.G;
        }

        public int GetHeuristicGoalCount(SingleAgentState state)
        {
            var agentGoals = Goals[state.Agent];
            var counter = agentGoals.Count(goal => state.Map[goal.Row][goal.Column] == goal.Item);
            return agentGoals.Count - counter;
        }

        public int GetHeuristicMaxManhattenDist(SingleAgentState state)
        {
            var agentGoals = Goals[state.Agent];
            var maxDist = 0;
            for (var r = 0; r < state.Map.Count; r++)
            {
                var row = state.Map[r];
                for (var c = 0; c < row.Count; c++)
                {
                    var item = row[c];
                    foreach (var goal in agentGoals)
                    {
                        if (goal.Item == item)
                        {
                            var dist = Math.Abs(r - goal.Row) + Math.Abs(c - goal.Column);
                            maxDist = Math.Max(maxDist, dist);
                        }
                    }
                }
            }

            return maxDist;
        }
    }
}