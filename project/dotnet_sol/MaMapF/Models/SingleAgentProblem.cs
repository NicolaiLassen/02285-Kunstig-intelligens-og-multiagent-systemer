using System.Collections.Generic;
using System.Linq;

namespace MaMapF.Models
{
    public class SingleAgentProblem
    {
        public char AgentName { get; set; }
        public SingleAgentState InitialState { get; set; }
        public List<MapItem> Goals { get; set; } = new List<MapItem>();
        public List<Position> WallModifications { get; set; } = new List<Position>();
        public List<MapItem> BoxModifications { get; set; } = new List<MapItem>();

        public void Reset()
        {
            foreach (var position in WallModifications)
            {
                InitialState.Walls.Remove($"{position.Row},{position.Column}");
            }

            foreach (var box in BoxModifications)
            {
                InitialState.Boxes.Add(box);
            }

            Goals = new List<MapItem>();
            WallModifications = new List<Position>();
            BoxModifications = new List<MapItem>();
        }

        public override string ToString()
        {
            var goalString = string.Join("\n", Goals.Select(g => g.ToString()));
            return $"SingleAgentProblem {AgentName}\n" +
                   $"{InitialState}" +
                   $"GOALS\n{goalString}\n";
        }
    }
}