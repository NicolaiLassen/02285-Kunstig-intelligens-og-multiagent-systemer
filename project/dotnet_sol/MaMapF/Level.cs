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

        
        
    }
}