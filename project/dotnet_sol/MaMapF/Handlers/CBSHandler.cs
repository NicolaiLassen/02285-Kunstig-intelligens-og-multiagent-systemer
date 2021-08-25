using System.Collections.Generic;
using System.Linq;
using MaMapF.Models;
using Priority_Queue;

namespace MaMapF
{
    public class CBSHandler
    {
        public Level Level;
        public LowLevelSearch LowLevelSearch { get; }

        public CBSHandler(Level level)
        {
            Level = level;
            LowLevelSearch = new LowLevelSearch
            {
                Level = level
            };
        }

        public Dictionary<char, List<SingleAgentState>> Search()
        {
            var open = new SimplePriorityQueue<Node>();
            var solutions = new Dictionary<char, List<SingleAgentState>>();
            foreach (var agent in Level.Agents)
            {
                solutions[agent] = LowLevelSearch.GetSingleAgentPlan(agent, new List<Constraint>());
            }

            var initialNode = new Node
            {
                Solutions = solutions
            };

            open.Enqueue(initialNode, initialNode.Cost);

            while (open.Count != 0)
            {
                var p = open.Dequeue();
                return p.Solutions;
                var conflict = GetConflict(p);
                if (conflict == null)
                {
                    return p.Solutions;
                }

                var agents = new List<char>
                {
                    conflict.AgentA,
                    conflict.AgentB
                };

                foreach (var agent in agents)
                {
                    var c = p.Copy();
                    var constraint = GetConstraint(agent, conflict);

                    // TODO - check shit

                    c.Constraints.Add(constraint);
                    // var solution = search

                    var solution = LowLevelSearch.GetSingleAgentPlan(agent, new List<Constraint>());
                    if (solution == null || solution == c.Solutions[agent])
                    {
                        continue;
                    }

                    c.Solutions[agent] = solution;
                    open.Enqueue(c, c.Cost);
                }
            }

            return null;
        }

        private static Conflict GetConflict(Node node)
        {
            var maxLength = node.Solutions.Max(solution => solution.Value.Count);
            var solutions = new Dictionary<char, List<SingleAgentState>>();
            foreach (var solutionsKey in node.Solutions.Keys)
            {
                var solutionsCount = node.Solutions[solutionsKey].Count;
                var solutionsCountDifference = maxLength - solutionsCount;
                solutions.Add(solutionsKey, node.Solutions[solutionsKey]);

                for (int i = 0; i < solutionsCountDifference; i++)
                {
                    var noOp =
                        LowLevelSearch.CreateNextState(solutions[solutionsKey][solutionsCount], Action.AllActions[0]);
                    solutions[solutionsKey].Add(noOp);
                }
            }

            return new Conflict();
        }

        private static Constraint GetConstraint(char agent, Conflict conflict)
        {
            return new Constraint();
        }
    }
}