using System.Collections.Generic;

namespace MaMapF.Models
{
    public class SingleAgentState
    {
        public SingleAgentState Parent { get; set; }
        public List<List<char>> Map { get; set; }

        public char Agent { get; set; }
        public Position AgentPosition { get; set; }
        public Action Action { get; set; }
        public int G { get; set; } // COST
        public int H { get; set; } // HEURISTIC
        public int F => G + H;

        public bool IsFree(Position position)
        {
            return Map[position.Row][position.Column] == ' ';
        }

        public bool IsBox(Position position)
        {
            var c = Map[position.Row][position.Column];
            return "ABCDEFGHIJKLMNOPQRSTUVWXYZ".Contains(c);
        }
    }
}