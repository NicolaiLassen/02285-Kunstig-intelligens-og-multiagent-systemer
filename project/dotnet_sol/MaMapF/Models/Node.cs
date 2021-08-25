﻿using System.Collections.Generic;
using System.Linq;

namespace MaMapF.Models
{
    public class Node
    {
        public List<Constraint> Constraints { get; set; }
        public Dictionary<string, SingleAgentState> Solutions { get; set; }
        public int Cost => Sic();
        private int Sic() => Solutions.Count(solution => solution.Value != null);
    }
}