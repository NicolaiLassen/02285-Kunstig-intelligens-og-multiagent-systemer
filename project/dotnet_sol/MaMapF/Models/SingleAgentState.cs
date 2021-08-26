using System.Collections.Generic;
using System.Linq;

namespace MaMapF.Models
{



    public class SingleAgentState
    {
        private int Hash = -1;
        public SingleAgentState Parent { get; set; }
        public List<List<char>> Map { get; set; }

        public char Agent { get; set; }
        public Position AgentPosition { get; set; }
        public List<MapItem> Boxes { get; set; }


        public Action Action { get; set; }
        public int G { get; set; } // COST
        public int H { get; set; } // HEURISTIC
        public int F => G + H;


        public override string ToString()
        {
            var info = $"Agent {Agent} at ({AgentPosition}) {Action}\nG: {G}, H: {H}, F: {F}";
            var map = string.Join("\n", Map.Select(row => string.Join("", row)));
            return $"{info}\n{map}\n";
        }

        public bool IsFree(Position position)
        {
            if (AgentPosition.Equals(position)) return false;
            if (IsBox(position)) return false;
            if (IsWall(position)) return false;
            return true;
        }

        public bool IsWall(Position position)
        {
            return Map[position.Row][position.Column] == '+';
        }

        public bool IsBox(Position position)
        {
            // return Boxes.Any(b => b.Position.Equals(position));
            return char.IsLetter(Map[position.Row][position.Column]);
        }


        public override int GetHashCode()
        {
            if (Hash != -1) return Hash;

            var prime = 31;
            var hash = prime + (AgentPosition.Row + 1) * 23 + (AgentPosition.Column + 1) * 29;
            hash = hash * prime + G * 13;
            hash = hash * prime + Map.Sum(row => row.Sum(c => c.GetHashCode()));
            Hash = hash;
            return Hash;
        }

        public override bool Equals(object obj)
        {
            if (obj == null) return false;
            if (!(obj is SingleAgentState other)) return false;
            if (G != other.G) return false;
            // if (AgentPosition.Row != other.AgentPosition.Row && AgentPosition.Column != other.AgentPosition.Column) return false;
            for (var row = 0; row < Map.Count; row++)
            {
                for (var col = 0; col < Map[row].Count; col++)
                {
                    if (Map[row][col] != other.Map[row][col])
                    {
                        return false;
                    }
                }
            }

            return true;
        }
    }
}