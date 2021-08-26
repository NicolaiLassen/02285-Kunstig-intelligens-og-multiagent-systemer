using System;
using System.Collections.Generic;
using System.Linq;
using MaMapF.Models;

namespace MaMapF
{
    public class LowLevelHeuristic
    {
        private Position AgentGoalPosition { get; set; }
        private List<Goal> BoxGoals { get; set; }

        public LowLevelHeuristic(List<Goal> goals)
        {
            var agentGoal = goals.FirstOrDefault(g => char.IsDigit(g.Item));
            if (agentGoal != null)
            {
                AgentGoalPosition = new Position(agentGoal.Row, agentGoal.Column);
            }

            BoxGoals = goals.Where(g => !char.IsDigit(g.Item)).ToList();
        }


        public int GetHeuristic(SingleAgentState state)
        {
            var h = 0;


            var emptyBoxGoals = BoxGoals.Where(g =>
                !state.Boxes.Any(b => b.Value == g.Item && b.Position.Equals(new Position(g.Row, g.Column)))).ToList();

            // Console.WriteLine($"emptyBoxGoals: {emptyBoxGoals.Count}");
            // if (!emptyBoxGoals.Any())
            // {
            //     Environment.Exit(0);
            // }

            if (emptyBoxGoals.Any())
            {
                // find box with shortest distance to goal
                var minBoxDistance = Int32.MaxValue;
                MapItem minBox = null;
                foreach (var boxGoal in emptyBoxGoals)
                {
                    foreach (var box in state.Boxes)
                    {
                        var distance = Math.Abs(boxGoal.Row - box.Position.Row) +
                                       Math.Abs(boxGoal.Column - box.Position.Column);
                        if (distance < minBoxDistance)
                        {
                            minBoxDistance = distance;
                            minBox = box;
                        }
                    }
                }

                // dont know
                if (minBox != null && minBoxDistance != Int32.MaxValue)
                {
                    // Add distance from closest box to non-taken goal
                    h += minBoxDistance;

                    // Add distance from agent to minBox
                    var agentDistanceToMinBox = Math.Abs(state.AgentPosition.Row - minBox.Position.Row) +
                                                Math.Abs(state.AgentPosition.Column - minBox.Position.Column);
                    h += agentDistanceToMinBox;

                }


            }


            // Add manhatten distance to agent goal
            // if (minBoxDistance == Int32.MaxValue)
            // {
            if (AgentGoalPosition != null)
            {
                var dist = Math.Abs(AgentGoalPosition.Row - state.AgentPosition.Row) +
                           Math.Abs(AgentGoalPosition.Column - state.AgentPosition.Column);
                h += dist;
            }
            // }


            return h;
        }
    }
}