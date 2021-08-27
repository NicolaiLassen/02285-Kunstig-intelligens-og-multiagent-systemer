using System;
using System.Collections.Generic;
using System.Linq;
using MaMapF.Models;

namespace MaMapF
{
    public class LowLevelHeuristic
    {
        private MapItem AgentGoal { get; }
        private List<MapItem> BoxGoals { get; }
        private List<Constraint> Constraints { get; }

        public LowLevelHeuristic(List<MapItem> goals, List<Constraint> constraints)
        {
            AgentGoal = goals.FirstOrDefault(g => char.IsDigit(g.Value));
            BoxGoals = goals.Where(g => !char.IsDigit(g.Value)).ToList();
            Constraints = constraints;
        }


        public int GetHeuristic(SingleAgentState state)
        {
            var h = 0;

            // Count future constraints that yields an alternative rute with an extra step
            h += Constraints.Count(constraint => constraint.Step > state.G);


            // Base heuristic on boxes before agent finish
            var emptyBoxGoals = BoxGoals.Where(goal => !state.Boxes.Any(box => box.Equals(goal))).ToList();
            var unusedBoxes = state.Boxes.Where(box => !BoxGoals.Any(goal => goal.Equals(box))).ToList();

            // Console.Error.WriteLine($"BoxGoals.Count: {BoxGoals.Count}");
            // Console.Error.WriteLine($"emptyBoxGoals: {emptyBoxGoals.Count}");
            // Console.Error.WriteLine($"state.Boxes.Count: {state.Boxes.Count}");
            // Console.Error.WriteLine($"unusedBoxes.Count: {unusedBoxes.Count}");
            //
            // Console.Error.WriteLine($"emptyBoxGoals.Any(): {emptyBoxGoals.Any()}");
            // Console.Error.WriteLine($"emptyBoxGoals.Count == unusedBoxes.Count: {emptyBoxGoals.Count == unusedBoxes.Count}");
            // Environment.Exit(0);

            // Does the format of the map include a box goal but no box
            if (emptyBoxGoals.Any())
            {
                // Find box with shortest distance to goal
                var minBoxDistance = Int32.MaxValue;
                MapItem minBox = null;
                foreach (var boxGoal in emptyBoxGoals)
                {
                    foreach (var box in unusedBoxes)
                    {
                        var distance = Position.Distance(boxGoal, box);
                        if (distance >= minBoxDistance) continue;
                        minBoxDistance = distance;
                        minBox = box;
                    }
                }


                // Add distance from closest box to non-taken goal
                h += minBoxDistance;

                // Add distance from agent to minBox
                var agentDistanceToMinBox = Position.Distance(state.Agent, minBox);
                h += agentDistanceToMinBox;
            }


            // Add distance from agent to agent goal
            if (AgentGoal != null)
            {
                h += Position.Distance(AgentGoal, state.Agent);
            }

            return h;
        }
    }
}