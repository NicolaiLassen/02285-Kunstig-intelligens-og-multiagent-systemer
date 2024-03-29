﻿using System;
using System.Collections.Generic;
using System.Linq;
using MaMapF.Models;
using Priority_Queue;
using Action = MaMapF.Models.Action;

namespace MaMapF.Handlers
{
    public class CBSHandler
    {
        public static Node Search(Dictionary<char, SingleAgentProblem> problems, List<char> solvedAgents, int pastSolutionLength)
        {
            // Create initial solutions for each agent
            var agents = problems.Keys.ToList();
            var solutions = agents.ToDictionary(agent => agent, agent =>
            {
                var problem = problems[agent];
                return SingleAgentSearchHandler.Search(
                    problem,
                    new List<Constraint>()
                );
            });

            // Create priority queue and add the initial node
            var initialNode = new Node {Solutions = solutions};
            var open = new SimplePriorityQueue<Node>();
            open.Enqueue(initialNode, initialNode.Cost);

            while (open.Count != 0)
            {
                // Console.Error.WriteLine(open.Count);
                // Get the node with lowest cost
                var node = open.Dequeue();

                // If no solutions conflict then return the solutions
                var conflict = GetConflict(agents, node, solvedAgents, pastSolutionLength);

                if (conflict == null)
                {
                    return node;
                }

                foreach (var agent in new List<char> {conflict.AgentA, conflict.AgentB})
                {
                    // Create next cbs search node
                    var nextNode = node.Copy();

                    // Add constraint to next node
                    var constraint = GetConstraint(agent, conflict);
                    nextNode.Constraints.Add(constraint);

                    // Skip if the previous node already contains the new constraint 
                    if (node.Constraints.Contains(constraint))
                    {
                        continue;
                    }
                    
                    // Create agent solution with new constraint
                    var constraints = nextNode.Constraints.Where(c => c.Agent == agent).ToList();

                    var solution = SingleAgentSearchHandler.Search(problems[agent], constraints);

                    // Skip if agent solution is null
                    if (solution == null)
                    {
                        continue;
                    }

                    // Skip if agent solution is equal agent solution in previous node
                    if (solution == nextNode.Solutions[agent])
                    {
                        continue;
                    }

                    // Update agent solution and add node to queue
                    nextNode.Solutions[agent] = solution;
                    open.Enqueue(nextNode, nextNode.Cost);
                }
            }
            
            return null;
        }

        private static Conflict GetConflict(
            List<char> agents,
            Node node,
            List<char> solvedAgents,
            int pastSolutionLength
        )
        {
            // var maxLength = node.Solutions.Max(solution => solution.Value.Count);
            var unsolvedMinLength = node.Solutions
                .Where(s => !solvedAgents.Contains(s.Key))
                .Min(solution => solution.Value.Count);

            var maxSteps = unsolvedMinLength;
            var solutions = agents
                .ToDictionary(agent => agent, agent => node.Solutions[agent].Select(s => s).ToList());
            
            // Make all solutions same length as longest
            foreach (var agent in solutions.Keys)
            {
                var solutionLength = solutions[agent].Count;
                if (solutionLength >= maxSteps)
                {
                    continue;
                }

                // Fill agents with NoOp to handle future constraints in same delegation
                var solutionLengthDiff = maxSteps - solutionLength;
                var nextState = solutions[agent].Last();
                for (var i = 0; i < solutionLengthDiff; i++)
                {
                    nextState = SingleAgentSearchHandler.CreateNextState(nextState, Action.NoOp);
                    solutions[agent].Add(nextState);
                }
            }

            for (var step = pastSolutionLength; step < maxSteps; step++)
            {
                for (var a0i = 0; a0i < agents.Count; a0i++)
                {
                    for (var a1i = a0i + 1; a1i < agents.Count; a1i++)
                    {
                        var a0 = agents[a0i];
                        var a1 = agents[a1i];
                        var a0s = solutions[a0];
                        var a1s = solutions[a1];

                        // It not a conflict if its myself
                        if (a0s[0].Color == a1s[0].Color)
                        {
                            continue;
                        }

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
                                        Type = "follow",
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
                                        Type = "follow",
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
            // Position that should be untouched
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