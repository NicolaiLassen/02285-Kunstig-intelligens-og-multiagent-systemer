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

        public Dictionary<char, SingleAgentState> AgentInitialStates { get; set; }

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

            foreach (var agent in agents)
            {
                AgentInitialStates[agent] = CreateAgentInitialState(agent);
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

        public float GetHeuristic(SingleAgentState state)
        {
            var f = state.G;
            var boxGoals = Goals[state.Agent].Where(goal => goal.Item != state.Agent);


            // Add max box distance

            var maxBoxDistance = 0;
            for (var row = 0; row < state.Map.Count; row++)
            {
                for (var col = 0; col < state.Map[row].Count; col++)
                {
                    var c = state.Map[row][col];
                    foreach (var boxGoal in boxGoals)
                    {
                        if (c == boxGoal.Item)
                        {
                            var dist = Math.Abs(row - boxGoal.Row) + Math.Abs(col - boxGoal.Column);
                            maxBoxDistance = Math.Max(maxBoxDistance, dist);
                        }
                    }
                }
            }


            // Add manhatten distance to agent goal
            var agentGoal = Goals[state.Agent].FirstOrDefault(goal => goal.Item == state.Agent);
            if (agentGoal != null)
            {
                var dist = Math.Abs(agentGoal.Row - state.AgentPosition.Row) +
                           Math.Abs(agentGoal.Column - state.AgentPosition.Column);
                f += dist;
            }

            return f;

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