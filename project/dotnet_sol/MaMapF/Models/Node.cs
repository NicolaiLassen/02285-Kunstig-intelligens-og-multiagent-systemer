using System.Collections.Generic;
using System.Linq;

namespace MaMapF.Models
{
    public class Node
    {
        public List<Constraint> Constraints { get; set; }
        public Dictionary<char, List<SingleAgentState>> Solutions { get; set; }
        public int Cost => Sic();

        private int Sic() => Solutions.Sum(solution => solution.Value.Count);

        public Node Copy()
        {
            return new Node
            {
                Constraints = Constraints,
                Solutions = Solutions,
            };
        }
    }
}