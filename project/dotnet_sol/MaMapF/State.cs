using System.Collections.Generic;
using MaMapF.Models;
using Action = System.Action;

namespace MaMapF
{
    public class State
    {
        public State Parent { get; set; }

        public List<List<char>> Map { get; set; }

        public Action Action { get; set; }
        public char Agent { get; set; }
        public Position AgentPosition { get; set; }
        public int g { get; set; }

        public List<State> GetSolution()
        {
            var solution = new List<State>();
            var state = this;

            while (state.Action != null)
            {
                solution.Insert(0, state);
                state = state.Parent;
            }

            solution.Insert(0, state);
            return solution;
        }
    }
}