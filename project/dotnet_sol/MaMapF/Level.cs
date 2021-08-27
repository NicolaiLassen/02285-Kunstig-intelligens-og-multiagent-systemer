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
        public Dictionary<char, List<MapItem>> Goals { get; set; }

        public Dictionary<char, SingleAgentState> AgentInitialStates { get; set; }


        public Level(
            Dictionary<char, string> colors,
            List<char> agents,
            List<List<char>> initialMap,
            List<List<char>> goalMap,
            Dictionary<char, List<MapItem>> goals
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
            var map = InitialMap.Select(r => r.Select(c => c).ToList()).ToList();
            var agentColor = Colors[agent];
            var agentPosition = new Position();
            var boxes = new List<MapItem>();
            var walls = new HashSet<string>();


            for (var row = 0; row < InitialMap.Count; row++)
            {
                for (var col = 0; col < InitialMap[row].Count; col++)
                {
                    var c = InitialMap[row][col];
                    if (c == ' ')
                    {
                        continue;
                    }

                    if (c == '+')
                    {
                        walls.Add($"{row},{col}");
                        continue;
                    }


                    if (c == agent)
                    {
                        agentPosition = new Position(row, col);
                        continue;
                    }

                    // Remove other agents
                    if (char.IsDigit(c))
                    {
                        map[row][col] = ' ';
                    }

                    if (!char.IsLetter(c)) continue;

                    // Remove other agent boxes
                    if (Colors[c] != agentColor)
                    {
                        map[row][col] = ' ';
                    }

                    if (Colors[c] == agentColor)
                    {
                        boxes.Add(new MapItem(c, new Position(row, col)));
                    }
                }
            }

            return new SingleAgentState
            {
                G = 0,
                // Map = map,
                Agent = new MapItem(agent, agentPosition),
                Boxes = boxes,
                Walls = walls,
            };
        }
    }
}