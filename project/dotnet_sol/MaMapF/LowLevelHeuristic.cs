using System;
using System.Collections.Generic;
using System.Linq;
using MaMapF.Models;

namespace MaMapF
{
    public class LowLevelHeuristic
    {
        public Position AgentGoalPosition { get; set; }
        public List<Goal> BoxGoals { get; set; } = new List<Goal>();

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


            // var emptyBoxGoals = BoxGoals.Where(g => state.IsBox(new Position(g.Row, g.Column)));


            // Find box closest to goal position not already taken
            var minBoxPosition = new Position();
            var minBoxDistance = Int32.MaxValue;
            for (var row = 0; row < state.Map.Count; row++)
            {
                for (var col = 0; col < state.Map[row].Count; col++)
                {
                    var c = state.Map[row][col];
                    foreach (var boxGoal in BoxGoals)
                    {
                        // skip taken goals
                        if (state.Map[boxGoal.Row][boxGoal.Column] == boxGoal.Item)
                        {
                            continue;
                        }

                        if (c == boxGoal.Item)
                        {
                            var dist = Math.Abs(row - boxGoal.Row) + Math.Abs(col - boxGoal.Column);
                            if (dist < minBoxDistance)
                            {
                                minBoxPosition = new Position(row, col);
                                minBoxDistance = dist;
                            }
                        }
                    }
                }
            }

            // Add distance from closest box to non-taken goal
            h += minBoxDistance;

            // Add distance from agent to minBox
            var agentDistanceToMinBox = Math.Abs(state.AgentPosition.Row - minBoxPosition.Row) +
                                        Math.Abs(state.AgentPosition.Column - minBoxPosition.Column);
            h += agentDistanceToMinBox;


            // Add manhatten distance to agent goal
            if (minBoxDistance == Int32.MaxValue)
            {
                if (AgentGoalPosition != null)
                {
                    var dist = Math.Abs(AgentGoalPosition.Row - state.AgentPosition.Row) +
                               Math.Abs(AgentGoalPosition.Column - state.AgentPosition.Column);
                    h += dist;
                }
            }


            return h;
        }
    }
}