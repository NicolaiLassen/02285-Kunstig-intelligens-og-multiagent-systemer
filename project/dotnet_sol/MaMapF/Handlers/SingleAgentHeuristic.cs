using System.Collections.Generic;
using System.Linq;
using MaMapF.Models;

namespace MaMapF.Handlers
{
    public class SingleAgentHeuristic
    {
        private MapItem AgentGoal { get; }
        private List<MapItem> BoxGoals { get; }
        private List<Constraint> Constraints { get; }

        public SingleAgentHeuristic(List<MapItem> goals, List<Constraint> constraints)
        {
            AgentGoal = goals.FirstOrDefault(g => char.IsDigit(g.Value));
            BoxGoals = goals.Where(g => !char.IsDigit(g.Value)).ToList();
            Constraints = constraints;
        }

        public int GetHeuristic(SingleAgentProblem problem, SingleAgentState state)
        {
            // Add future constraint count since every constraint yields 1 extra step
            var futureConstraints = Constraints.Where(constraint => constraint.Step > state.G).ToList();

            // Add distance to boxes included in future constraints
            var h = (from constraint in futureConstraints
                from box in state.Boxes
                where constraint.Position.Equals(box.Position)
                select Position.Distance(state.Agent, box)).Sum();

            if (problem.Type == SingleAgentProblemType.BoxToGoal)
            {
                var currentSelectedBoxPosition =
                    state.Boxes.FirstOrDefault(b => b.UID == problem.SelectedBox.UID);

                // Add distance from agent to selectedBox
                h += Position.Distance(state.Agent, currentSelectedBoxPosition);

                // Add distance from selected box to selected goal
                h += Position.Distance(currentSelectedBoxPosition, problem.SelectedBoxGoal);
            }

            // Add distance from agent to agent goal position
            if (AgentGoal != null)
            {
                h += Position.Distance(AgentGoal, state.Agent);
            }

            return h;
        }
    }
}