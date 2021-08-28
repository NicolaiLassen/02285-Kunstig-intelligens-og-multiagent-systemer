using System;
using System.Collections.Generic;

namespace MaMapF.Models
{
    public class Delegate
    {
        public Dictionary<char, SingleAgentState> NextInitialStates { get; set; }

        public Dictionary<char, List<Position>> WallModifications { get; set; } =
            new Dictionary<char, List<Position>>();

        public Dictionary<char, List<MapItem>> BoxModifications { get; set; } =
            new Dictionary<char, List<MapItem>>();

        public List<string> UsedBoxes { get; } = new List<string>();

        public Dictionary<char, List<MapItem>> Goals { get; set; }

        public void ResetNextStateDelegation()
        {
            var resetNext = new Dictionary<char, SingleAgentState>(NextInitialStates);
            foreach (var agent in WallModifications.Keys)
            {
                foreach (var position in WallModifications[agent])
                {
                    resetNext[agent].Walls.Remove($"{position.Row},{position.Column}");
                }
                
                foreach (var box in BoxModifications[agent])
                {
                    resetNext[agent].Boxes.Add(box);
                }
            }
        }
    }
}