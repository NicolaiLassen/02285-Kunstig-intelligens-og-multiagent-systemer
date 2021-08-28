using System.Collections.Generic;
using System.Linq;

namespace MaMapF.Models
{
    public class SingleAgentProblem
    {
        public SingleAgentState InitialState { get; set; }
        public List<MapItem> Goals { get; set; }
        public List<MapItem> BoxMods { get; set; }
        public List<Position> WallMods { get; set; }

        public SingleAgentProblem(SingleAgentState initialState)
        {
            InitialState = initialState;
            Goals = new List<MapItem>();
            WallMods = new List<Position>();
            BoxMods = new List<MapItem>();
        }


        public void AddBoxMod(MapItem box)
        {
            BoxMods.Add(box);
            WallMods.Add(box.Position);
            InitialState.Walls.Add($"{box.Position.Row},{box.Position.Column}");
            InitialState.Boxes = InitialState.Boxes.Where(b => !b.Equals(box)).ToList();
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
            BoxMods = new List<MapItem>();
            WallMods = new List<Position>();
        }

        public override string ToString()
        {
            var goalString = string.Join("\n", Goals.Select(g => g.ToString()));
            return $"SingleAgentProblem\n" +
                   $"{InitialState}" +
                   $"GOALS\n{goalString}\n";
        }
    }
}