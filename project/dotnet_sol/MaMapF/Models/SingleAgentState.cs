using System.Collections.Generic;
using System.Linq;

namespace MaMapF.Models
{
    public class SingleAgentState
    {
        private int Hash = -1;
        public SingleAgentState Parent { get; set; }

        public HashSet<string> Walls { get; set; }
        // public List<List<char>> Map { get; set; }

        public MapItem Agent { get; set; }
        public List<MapItem> Boxes { get; set; }

        public char AgentName => Agent.Value;
        
        public Action Action { get; set; }
        public int G { get; set; } // COST
        public int H { get; set; } // HEURISTIC
        public int F => G + H;


        public List<MapItem> AllMapItems => Boxes.Concat(new[] {Agent}).ToList();
        public List<Position> AllPositions => AllMapItems.Select(i => i.Position).ToList();

        public override string ToString()
        {
            var info = $"Agent {AgentName} at ({Agent.Position}) {Action}\nG: {G}, H: {H}, F: {F}";
            return $"{info}\n";
            // var map = string.Join("\n", Map.Select(row => string.Join("", row)));
            // var map = "";
            // return $"{info}\n{map}\n";
        }

        public bool IsFree(Position position)
        {
            if (Agent.Position.Equals(position)) return false;
            if (IsBox(position)) return false;
            if (IsWall(position)) return false;
            return true;
        }

        public bool IsWall(Position position)
        {
            return Walls.Contains($"{position.Row},{position.Column}");
        }

        public bool IsBox(Position position)
        {
            return Boxes.Any(b => b.Position.Equals(position));
        }


        public override int GetHashCode()
        {
            if (Hash != -1) return Hash;

            var prime = 31;
            var hash = prime * 1;
            hash = hash * prime + G * 23;
            hash = hash * prime * AllMapItems.Sum(item =>
                item.Value.GetHashCode() + item.Position.Row * 11 + item.Position.Column * 13);
            Hash = hash;
            return Hash;
        }

        public override bool Equals(object obj)
        {
            if (obj == null) return false;
            if (!(obj is SingleAgentState other)) return false;

            // If the time is different
            if (G != other.G) return false;

            // If my agent is the same as the other state
            if (!Agent.Equals(other.Agent)) return false;

            // If the other state has all the boxes that I do
            return Boxes.All(b => other.Boxes.Any(b.Equals));
        }
    }
}