﻿using System.Collections.Generic;
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
        public Action Action { get; set; }
        public int G { get; set; } // COST
        public int H { get; set; } // HEURISTIC
        public int F => G + H;

        public override int GetHashCode()
        {
            if (Hash != -1) return Hash;

            var prime = 31;
            var hash = prime + AgentPosition.Row * 23 + AgentPosition.Column * 29 +
                       (int) char.GetNumericValue(Agent);
            hash = hash * prime + Map.Sum(row => row.GetHashCode());
            Hash = hash;

            return Hash;
        }

        public override bool Equals(object obj)
        {
            if (obj == null) return false;
            if (!(obj is SingleAgentState other)) return false;
            if (G != other.G) return false;
            if (AgentPosition != other.AgentPosition) return false;

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