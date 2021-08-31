using System.Collections.Generic;
using System.Linq;

namespace MaMapF.Models
{
    public class Node
    {
        private int Hash = -1;
        public List<Constraint> Constraints { get; set; } = new List<Constraint>();

        public Dictionary<char, List<SingleAgentState>> Solutions { get; set; } =
            new Dictionary<char, List<SingleAgentState>>();

        // https://arxiv.org/pdf/2006.03280.pdf
        public int Cost => Solutions.Max(solution => solution.Value.Count) + Constraints.Count;
        
        public Node Copy()
        {
            return new Node
            {
                Constraints = Constraints.Select(c => c.Copy()).ToList(),
                Solutions = Solutions.ToDictionary(
                    agent => agent.Key,
                    solution =>
                        solution.Value.Select(s => s).ToList()),
            };
        }

        public override bool Equals(object obj)
        {
            if (obj == null) return false;
            if (!(obj is Node other)) return false;
            return Constraints.All(c => other.Constraints.Any(c.Equals));
        }

        public override int GetHashCode()
        {
            if (Hash != -1) return Hash;
            var prime = 31;
            var hash = prime * 1;
            hash = hash + Constraints.Sum(item => item.GetHashCode() % 257);
            Hash = hash;
            return Hash;
        }

        public override string ToString()
        {
            var solutionString = string.Join("\n--------\n",
                Solutions.Values.Select(val => string.Join("\n", val)).ToList());
            return $"SOLUTION\nCost: {Cost}\n{solutionString}";
        }
    }
}