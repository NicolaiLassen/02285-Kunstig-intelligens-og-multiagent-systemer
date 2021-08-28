using System.Collections.Generic;
using System.Linq;

namespace MaMapF.Models
{
    public class SingleAgentProblem
    {
        public char AgentName { get; set; }
        public SingleAgentState InitialState { get; set; }
        public List<MapItem> Goals { get; set; } = new List<MapItem>();
        public List<Position> WallMods { get; set; } = new List<Position>();
        public List<MapItem> BoxMods { get; set; } = new List<MapItem>();

        public void AddBoxMod(MapItem box)
        {
            BoxMods.Add(box);
            WallMods.Add(box.Position);
            InitialState.Walls.Add($"{box.Position.Row},{box.Position.Column}");
        }

        public void Reset()
        {
            foreach (var position in WallMods)
            {
                InitialState.Walls.Remove($"{position.Row},{position.Column}");
            }

            foreach (var box in BoxMods)
            {
                InitialState.Boxes.Add(box);
            }

            Goals = new List<MapItem>();
            WallMods = new List<Position>();
            BoxMods = new List<MapItem>();
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