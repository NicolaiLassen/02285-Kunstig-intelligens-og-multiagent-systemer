using System;
using System.Collections.Generic;
using System.Linq;
using MaMapF.Models;
using Priority_Queue;
using Action = MaMapF.Models.Action;

namespace MaMapF
{
    public class CBSHandler
    {
        public Level Level { get; }
        public LowLevelSearch LowLevelSearch { get; }

        public CBSHandler(Level level)
        {
            Level = level;
            LowLevelSearch = new LowLevelSearch(level);
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

                var conflict = GetConflict(p);
                if (conflict == null)
                {
                    foreach (var singleAgentStates in p.Solutions.Values)
                    {
                        foreach (var singleAgentState in singleAgentStates)
                        {
                            Console.Error.WriteLine(singleAgentState);
                        }
                    }

                    return p.Solutions;
                }

                // Console.Error.WriteLine("Conflict");
                // Console.Error.WriteLine(conflict);
                // Environment.Exit(0);

                var agents = new List<char>
                {
                    conflict.AgentA,
                    conflict.AgentB
                };

                foreach (var agent in agents)
                {
                    var c = p.Copy();
                    var constraint = GetConstraint(agent, conflict);
                    if (constraint == null)
                    {
                        Console.Error.WriteLine("Constraint == null");
                        Environment.Exit(0);
                    }

                    // TODO - check shit

                    c.Constraints.Add(constraint);

                    var solution = LowLevelSearch.GetSingleAgentPlan(agent, p.Constraints);
                    if (solution == null)
                    {
                        Console.Error.WriteLine("456645645645645645");
                        Environment.Exit(0);
                    }

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

        private Conflict GetConflict(Node node)
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

            for (var step = 1; step < maxLength; step++)
            {
                for (var a0i = 0; a0i < Level.Agents.Count; a0i++)
                {
                    for (var a1i = a0i + 1; a1i < Level.Agents.Count; a1i++)
                    {
                        var a0 = Level.Agents[a0i];
                        var a1 = Level.Agents[a1i];

                        var agent0S = node.Solutions[a0];
                        var agent1S = node.Solutions[a1];

                        // CONFLICT if agent 1 and agent 2 is at same position
                        if (agent0S[step].AgentPosition.Equals(agent1S[step].AgentPosition))
                        {
                            return new Conflict
                            {
                                Type = "position",
                                AgentA = a0,
                                AgentB = a1,
                                Position = agent0S[step].AgentPosition,
                                Step = step
                            };
                        }
                    }
                }
            }

            return null;
        }

        private static Constraint GetConstraint(char agent, Conflict conflict)
        {
            if (conflict.Type == "position")
            {
                return new Constraint
                {
                    Agent = agent,
                    Position = conflict.Position,
                    Step = conflict.Step,
                    Conflict = conflict
                };
            }

            return null;
        }
    }
}