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
    }
}