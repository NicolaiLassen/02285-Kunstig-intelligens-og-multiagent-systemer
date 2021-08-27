using System.Collections.Generic;

namespace MaMapF.Models
{
    public class GoalDelegate
    {
        public Dictionary<char, SingleAgentState> NextInitialStates { get; set; }
        public Dictionary<char, List<MapItem>> Goals { get; set; }
    }
}