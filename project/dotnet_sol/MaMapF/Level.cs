using System.Collections.Generic;
using MaMapF.Models;

namespace MaMapF
{
    public class Level
    {
        public Dictionary<char, char> Colors { get; set; }
        public int AgentCount { get; set; }

        public List<List<char>> InitialMatrix { get; set; }
        public List<List<char>> GoalMatrix { get; set; }

        public State GetClientInitialState(char agent)
        {
            
            return new State();
        }
    }
}