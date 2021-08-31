using System.Collections.Generic;
using System.Linq;

namespace MaMapF.Models
{
    public class SingleAgentState
    {
        private int Hash = -1;
        public int ROWS { get; set; }
        public int COLS { get; set; }
        public string Color { get; set; }

        public Constraint PastConstraint { get; set; }

        public char AgentName => Agent.Value;
        public MapItem Agent { get; set; }
        public List<MapItem> Boxes { get; set; }
        public HashSet<string> Walls { get; set; }
        public List<MapItem> BoxWalls { get; set; }
        public SingleAgentState Parent { get; set; }
        public Action Action { get; set; }
        public int G { get; set; } // COST
        public int H { get; set; } // HEURISTIC
        public int F => G + H;

        public List<MapItem> AllMapItems => BoxWalls.Concat(Boxes.Concat(new[] {Agent}).ToList()).ToList();
        public List<Position> AllPositions => AllMapItems.Select(i => i.Position).ToList();

        public override string ToString()
        {
            var info = $"Agent {AgentName} at ({Agent.Position}) {Action}\n" +
                       $"G: {G}, H: {H}, F: {F}\n";

            var map = new string(' ', ROWS).Select(r => new string(' ', COLS).Select(e => e).ToList()).ToList();
            for (int r = 0; r < ROWS; r++)
            {
                // map[r] ??= new List<char>(COLS);
                for (int c = 0; c < COLS; c++)
                {
                    var pos = new Position(r, c);
                    if (IsWall(pos))
                    {
                        map[r][c] = '+';
                        continue;
                    }

                    if (IsAgent(pos))
                    {
                        map[r][c] = AgentName;
                        continue;
                    }

                    var box = Boxes.FirstOrDefault(b => pos.Equals(b.Position));
                    if (box != null)
                    {
                        map[r][c] = box.Value;
                        continue;
                    }

                    map[r][c] = ' ';
                }
            }

            var mapString = string.Join("\n", map.Select(row => string.Join("", row)));


            return $"{info}{mapString}";
            // var map = "";
            // return $"{info}\n{map}\n";
        }

        public bool IsFree(Position position)
        {
            if (IsAgent(position)) return false;
            if (IsBox(position)) return false;
            if (IsWall(position)) return false;
            return true;
        }

        public bool IsAgent(Position position)
        {
            return Agent.Position.Equals(position);
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
            if (PastConstraint != null)
            {
                hash = hash + PastConstraint.GetHashCode();
            }

            // hash = hash * prime + G * 23;
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
            if (PastConstraint != null && other.PastConstraint != null)
            {
                if (!PastConstraint.Equals(other.PastConstraint)) return false;
            }
            // if (G != other.G) return false;

            // If my agent is the same as the other state
            if (!Agent.Equals(other.Agent)) return false;

            // If the other state has all the boxes that I do
            return Boxes.All(b => other.Boxes.Any(b.Equals));
        }
    }
}