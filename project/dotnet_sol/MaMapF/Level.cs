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

        public List<List<char>> InitialMatrix { get; set; }
        public List<List<char>> GoalMatrix { get; set; }

        public Dictionary<char, List<Goal>> Goals { get; set; }

        public SingleAgentState GetAgentInitialState(char agent)
        {
            var map = new List<List<char>>(InitialMatrix);
            var agentColor = Colors[agent];
            var agentPosition = new Position();

            for (var row = 0; row < InitialMatrix.Count; row++)
            {
                for (var col = 0; col < InitialMatrix[row].Count; col++)
                {
                    var c = InitialMatrix[row][col];
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

        public bool IsAgentGoalState(SingleAgentState state)
        {
            var agentGoals = Goals[state.Agent];
            var counter = agentGoals.Count(agentGoal => state.Map[agentGoal.Row][agentGoal.Column] == agentGoal.Item);
            return counter == Goals[state.Agent].Count;
        }

        public float GetHeuristic(SingleAgentState state)
        {
            // var h = GetHeuristicGoalCount(state);
            var h = GetHeuristicMaxManhattenDist(state);
            return h + state.G;
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