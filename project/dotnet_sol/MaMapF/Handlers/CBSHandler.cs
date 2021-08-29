using System.Collections.Generic;
using System.Linq;
using MaMapF.Models;
using Priority_Queue;

namespace MaMapF.Handlers
{
    public class CBSHandler
    {
        public static Node Search(
            Dictionary<char, SingleAgentProblem> problems
        )
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
                // Get the node with lowest cost
                var p = open.Dequeue();

                // If no solutions conflict then return the solutions
                var conflict = GetConflict(agents, p);
                if (conflict == null)
                {
                    return p;
                }

                foreach (var agent in new List<char> {conflict.AgentA, conflict.AgentB})
                {
                    // Create next cbs search node
                    var nextNode = p.Copy();

                    // Add constraint to next node
                    var constraint = GetConstraint(agent, conflict);
                    nextNode.Constraints.Add(constraint);

                    // Skip if the previous node already contains the new constraint 
                    if (p.Constraints.Contains(constraint))
                    {
                        continue;
                    }

                    // Create agent solution with new constraint
                    var constraints = nextNode.Constraints.Where(c => c.Agent == agent).ToList();
                    var solution = SingleAgentSearchHandler.Search(problems[agent], constraints);

                    // Skip if agent solution is null
                    if (solution == null)
                    {
                        if (constraint.Agent == agent)
                        {
                            continue;
                        }

                        nextNode.Blocked = new Blocked
                        {
                            Agent = constraint.Agent,
                            Position = constraint.Position
                        };
                        return nextNode;
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

        private static Conflict GetConflict(List<char> agents, Node node)
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
                    var nextState = SingleAgentSearchHandler.CreateNextState(solutionGoalState, Action.NoOp);
                    solutions[agent].Add(nextState);
                }
            }

            for (var step = 1; step < maxLength; step++)
            {
                for (var a0i = 0; a0i < agents.Count; a0i++)
                {
                    for (var a1i = a0i + 1; a1i < agents.Count; a1i++)
                    {
                        var a0 = agents[a0i];
                        var a1 = agents[a1i];
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

            // // Agent is follower
            // if (conflict.Type == "follow" && agent == conflict.AgentA)
            // {
            //     return new Constraint
            //     {
            //         Agent = agent,
            //         Position = conflict.Position,
            //         Step = conflict.Step,
            //     };
            // }
            //
            // // Agent is leader
            // if (conflict.Type == "follow" && agent == conflict.AgentB)
            // {
            //     return new Constraint
            //     {
            //         Agent = agent,
            //         Position = conflict.Position,
            //         Step = conflict.Step - 1,
            //     };
            // }

            return null;
        }
    }
}