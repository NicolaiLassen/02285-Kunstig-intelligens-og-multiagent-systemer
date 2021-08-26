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
                solutions[agent] = LowLevelSearch.GetSingleAgentPlan(agent, new List<Constraint>()
                {
                    new Constraint()
                    {
                        Agent = '0',
                        Position = new Position(2, 1),
                        Step = 1,
                    }
                });
            }

            var initialNode = new Node
            {
                Solutions = solutions
            };

            open.Enqueue(initialNode, initialNode.Cost);

            while (open.Count != 0)
            {
                var p = open.Dequeue();

                Console.Error.WriteLine($"OPEN: {open.Count}");

                var conflict = GetConflict(p);
                if (conflict == null)
                {
                    // PRINT SOLUTION
                    // foreach (var singleAgentStates in p.Solutions.Values)
                    // {
                    //     foreach (var singleAgentState in singleAgentStates)
                    //     {
                    //         Console.Error.WriteLine(singleAgentState);
                    //     }
                    // }

                    return p.Solutions;
                }

                // Console.Error.WriteLine("Conflict");
                // Console.Error.WriteLine(conflict);
                // Environment.Exit(0);

                foreach (var agent in new List<char> {conflict.AgentA, conflict.AgentB})
                {
                    var nextNode = p.Copy();

                    var constraint = GetConstraint(agent, conflict);
                    // if (constraint == null)
                    // {
                    //     Console.Error.WriteLine("Constraint == null");
                    //     Environment.Exit(0);
                    // }

                    // TODO - check shit

                    nextNode.Constraints.Add(constraint);

                    var agentConstraints = p.Constraints.Where(c => c.Agent == agent).ToList();
                    var solution = LowLevelSearch.GetSingleAgentPlan(agent, agentConstraints);
                    if (solution == null || solution == nextNode.Solutions[agent])
                    {
                        continue;
                    }

                    nextNode.Solutions[agent] = solution;
                    open.Enqueue(nextNode, nextNode.Cost);
                }
            }

            return null;
        }

        private Conflict GetConflict(Node node)
        {
            var maxLength = node.Solutions.Max(solution => solution.Value.Count);
            var solutions = new Dictionary<char, List<SingleAgentState>>(node.Solutions);

            foreach (var agent in solutions.Keys)
            {
                var solutionLength = node.Solutions[agent].Count;
                var solutionLengthDiff = maxLength - solutionLength;
                var solutionGoalState = solutions[agent][solutionLength - 1];

                for (int i = 0; i < solutionLengthDiff; i++)
                {
                    var nextState = LowLevelSearch.CreateNextState(solutionGoalState, Action.NoOp);
                    solutions[agent].Add(nextState);
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
                };
            }

            return null;
        }
    }
}