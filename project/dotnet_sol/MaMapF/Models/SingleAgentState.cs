namespace MaMapF.Models
{
    public class SingleAgentState
    {
        public SingleAgentState Parent { get; set; }
        public Action Action { get; set; }

        public Position AgentPosition { get; set; }


        public int G { get; set; } // COST
        public int H { get; set; } // HEURISTIC
        public int F => G + H;
        
        
        
        
        
    }
}