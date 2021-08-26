﻿using System.Collections.Generic;
using System.Linq;
using MaMapF.Models;
using Priority_Queue;

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
                // Console.Error.WriteLine($"OPEN: {open.Count}");

                var p = open.Dequeue();
                // Console.Error.WriteLine($"p.Constraints.Count: {p.Constraints.Count}");
                // p.Constraints.ForEach(c => Console.Error.WriteLine(c));
                // Console.Error.WriteLine("");

                var conflict = GetConflict(p);
                // Console.Error.WriteLine(conflict);
                if (conflict == null)
                {
                    return p.Solutions;
                }

                foreach (var agent in new List<char> {conflict.AgentA, conflict.AgentB})
                {
                    var nextNode = p.Copy();

                    var constraint = GetConstraint(agent, conflict);
                    if (nextNode.Constraints.Contains(constraint))
                    {
                        continue;
                    }


                    nextNode.Constraints.Add(constraint);


                    var agentConstraints = nextNode.Constraints.Where(c => c.Agent == agent).ToList();
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

            // Make all solutions same length as longest
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
                        var a0s = node.Solutions[a0];
                        var a1s = node.Solutions[a1];


                        // Check that no positions are equal in current step
                        foreach (var a0p in a0s[step].AllPositions)
                        {
                            foreach (var a1p in a1s[step].AllPositions)
                            {
                                if (a0p.Equals(a1p))
                                {
                                    return new Conflict
                                    {
                                        Type = "position",
                                        AgentA = a0,
                                        AgentB = a1,
                                        Position = a0p,
                                        Step = step
                                    };
                                }
                            }
                        }

                        // Check that agent 0 does not move something to an agent 1 position
                        foreach (var a0p in a0s[step].AllPositions)
                        {
                            foreach (var a1p in a1s[step - 1].AllPositions)
                            {
                                if (a0p.Equals(a1p))
                                {
                                    return new Conflict
                                    {
                                        Type = "position",
                                        AgentA = a0,
                                        AgentB = a1,
                                        Position = a0p,
                                        Step = step
                                    };
                                }
                            }
                        }

                        // Check that agent 1 does not move something to an agent 0 position
                        foreach (var a0p in a0s[step - 1].AllPositions)
                        {
                            foreach (var a1p in a1s[step].AllPositions)
                            {
                                if (a0p.Equals(a1p))
                                {
                                    return new Conflict
                                    {
                                        Type = "position",
                                        AgentA = a1,
                                        AgentB = a0,
                                        Position = a1p,
                                        Step = step
                                    };
                                }
                            }
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

            // Agent is follower
            if (conflict.Type == "follow" && agent == conflict.AgentA)
            {
                return new Constraint
                {
                    Agent = agent,
                    Position = conflict.Position,
                    Step = conflict.Step,
                };
            }

            // Agent is leader
            if (conflict.Type == "follow" && agent == conflict.AgentB)
            {
                return new Constraint
                {
                    Agent = agent,
                    Position = conflict.Position,
                    Step = conflict.Step - 1,
                };
            }

            return null;
        }
    }
}