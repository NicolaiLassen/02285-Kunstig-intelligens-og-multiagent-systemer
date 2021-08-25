﻿using System.Collections.Generic;
using MaMapF.Models;
using Priority_Queue;

namespace MaMapF
{
    public class CBSHandler
    {
        public static Node Search(Dictionary<char, List<SingleAgentState>> solutions)
        {
            var open = new SimplePriorityQueue<Node>();

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
                    return p;
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
                    var solution = ' ';
                    if (solution == null || solution == c.Solutions)
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
            return new Conflict();
        }

        private static Constraint GetConstraint(char agent, Conflict conflict)
        {
            return new Constraint();
        }
    }
}