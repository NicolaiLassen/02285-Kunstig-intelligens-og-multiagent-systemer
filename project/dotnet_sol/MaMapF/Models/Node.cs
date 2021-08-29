using System.Collections.Generic;
using System.Linq;

namespace MaMapF.Models
{
    public class Node
    {
        public List<Constraint> Constraints { get; set; } = new List<Constraint>();

        public Dictionary<char, List<SingleAgentState>> Solutions { get; set; } =
            new Dictionary<char, List<SingleAgentState>>();


        // public int Cost => Solutions.Sum(solution => solution.Value.Count) + Constraints.Count;
        public int Cost => Solutions.Sum(solution => solution.Value.Count);

        public Constraint WallBoxConstraint { get; set; }

        public Node Copy()
        {
            return new Node
            {
                Constraints = Constraints.Select(c => c.Copy()).ToList(),
                Solutions = new Dictionary<char, List<SingleAgentState>>(Solutions),
            };
        }

        public override string ToString()
        {
            var solutionString = string.Join("\n--------\n",
                Solutions.Values.Select(val => string.Join("\n", val)).ToList());
            return $"SOLUTION\nCost: {Cost}\n{solutionString}";
        }
    }
}