using System.Collections.Generic;

namespace MaMapF.Models
{
    public class Level
    {
        public Dictionary<char, char> Colors { get; set; }
        public int AgentCount { get; set; }

        public List<List<char>> InitialMatrix { get; set; }
        public List<List<char>> GoalMatrix { get; set; }
    }
}