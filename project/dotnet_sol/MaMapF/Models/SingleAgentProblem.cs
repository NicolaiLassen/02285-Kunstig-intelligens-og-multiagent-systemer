using System.Collections.Generic;
using System.Linq;

namespace MaMapF.Models
{
    public enum SingleAgentProblemType
    {
        NULL,
        MoveBlock,
        AgentToBox,
        BoxToGoal,
        AgentToGoal,
    }

    public class SingleAgentProblem
    {
        public SingleAgentState InitialState { get; set; }
        public List<MapItem> Goals { get; set; }

        public List<Constraint> Constraints { get; set; }

        public MapItem SelectedBox { get; set; }
        public MapItem SelectedBoxGoal { get; set; }

        public SingleAgentProblemType Type { get; set; }


        public SingleAgentProblem(SingleAgentState initialState)
        {
            Type = SingleAgentProblemType.NULL;
            InitialState = initialState;
            Goals = new List<MapItem>();
            Constraints = new List<Constraint>();
            InitialState.BoxWalls = new List<MapItem>();
        }

        public void AddBoxMod(MapItem box)
        {
            InitialState.Walls.Add($"{box.Position.Row},{box.Position.Column}");
            InitialState.Boxes = InitialState.Boxes.Where(b => !b.Equals(box)).ToList();
            InitialState.BoxWalls = InitialState.BoxWalls.Concat(new[] {box}).ToList();
        }

        public void ResetMods()
        {
            foreach (var box in InitialState.BoxWalls)
            {
                InitialState.Walls.Remove($"{box.Position.Row},{box.Position.Column}");
                InitialState.Boxes.Add(box);
            }

            InitialState.BoxWalls = new List<MapItem>();
        }

        public override string ToString()
        {
            var goalString = string.Join("\n", Goals.Select(g => g.ToString()));
            return $"\n******************************\n" +
                   $"SingleAgentProblem\n" +
                   $"Type: {Type}\n" +
                   $"{InitialState}\n" +
                   $"GOALS\n" +
                   $"{goalString}\n" +
                   $"******************************\n";
        }
    }
}