using System;
using System.Collections.Generic;
using System.Linq;
using MaMapF.Models;

namespace MaMapF
{
    public class LowLevelHeuristic
    {
        private Position AgentGoalPosition { get; set; }
        private List<MapItem> BoxGoals { get; set; }

        public LowLevelHeuristic(List<MapItem> goals)
        {
            var agentGoal = goals.FirstOrDefault(g => char.IsDigit(g.Value));
            if (agentGoal != null)
            {
                AgentGoalPosition = agentGoal.Position;
            }

            BoxGoals = goals.Where(g => !char.IsDigit(g.Value)).ToList();
        }


        public int GetHeuristic(SingleAgentState state)
        {
            var h = 0;


            var emptyBoxGoals = BoxGoals.Where(g =>
                !state.Boxes.Any(b => b.Value == g.Value && b.Position.Equals(g.Position))).ToList();
            var hasBoxGoals = emptyBoxGoals.Any();

            // Console.WriteLine($"emptyBoxGoals: {emptyBoxGoals.Count}");
            // if (!emptyBoxGoals.Any())
            // {
            //     Environment.Exit(0);
            // }


            
            if (hasBoxGoals)
            {
                // find box with shortest distance to goal
                var minBoxDistance = Int32.MaxValue;
                MapItem minBox = null;
                foreach (var boxGoal in emptyBoxGoals)
                {
                    foreach (var box in state.Boxes)
                    {
                        var distance = Math.Abs(boxGoal.Position.Row - box.Position.Row) +
                                       Math.Abs(boxGoal.Position.Column - box.Position.Column);

                        Console.Error.WriteLine($"distance: {distance}");
                        if (distance >= minBoxDistance) continue;
                        minBoxDistance = distance;
                        minBox = box;
                    }
                }

                if (minBox == null)
                {
                    Console.Error.WriteLine("HOW");
                }

                // Add distance from closest box to non-taken goal
                h += minBoxDistance;

                // Add distance from agent to minBox
                var agentDistanceToMinBox = Math.Abs(state.AgentPosition.Row - minBox.Position.Row) +
                                            Math.Abs(state.AgentPosition.Column - minBox.Position.Column);
                h += agentDistanceToMinBox;
            }

            if (hasBoxGoals) return h;
            if (AgentGoalPosition == null) return h;
            
            var dist = Math.Abs(AgentGoalPosition.Row - state.AgentPosition.Row) +
                       Math.Abs(AgentGoalPosition.Column - state.AgentPosition.Column);
            
            return h + dist;
        }
    }
}