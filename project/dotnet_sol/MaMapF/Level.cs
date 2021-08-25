using System.Collections.Generic;
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

        public State GetClientInitialState(char agent)
        {
            return new State();
        }
    }
}