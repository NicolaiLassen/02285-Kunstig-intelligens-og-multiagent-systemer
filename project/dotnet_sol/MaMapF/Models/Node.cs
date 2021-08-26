using System.Collections.Generic;
using System.Linq;

namespace MaMapF.Models
{
    public class Node
    {
        public List<Constraint> Constraints { get; set; } = new List<Constraint>();

        public Dictionary<char, List<SingleAgentState>> Solutions { get; set; } =
            new Dictionary<char, List<SingleAgentState>>();

        public int Cost => Sic();

        private int Sic() => Solutions.Sum(solution => solution.Value.Count);

        public Node Copy()
        {
            return new Node
            {
                Constraints = Constraints.Select(c => c).ToList(),
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