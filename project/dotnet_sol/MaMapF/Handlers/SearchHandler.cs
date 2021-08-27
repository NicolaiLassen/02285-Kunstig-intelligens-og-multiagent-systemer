using System.Collections.Generic;
using System.Linq;
using MaMapF.Models;

namespace MaMapF.Handlers
{
    public class SearchHandler
    {
        private readonly Level _level;

        public SearchHandler(Level level)
        {
            _level = level;
        }

        private bool IsAllMainGoalsSolved(List<MapItem> solved)
        {
            var allSolved = true;
            foreach (var goal in _level.Goals.Values)
            {
                if (!goal.All(solved.Contains))
                {
                    allSolved = false;
                }
            }
            return allSolved;
        }

        public Dictionary<char, List<SingleAgentState>> Search()
        {
            var solved = new List<MapItem>();
            var solutions =
                _level.Agents.ToDictionary(levelAgent => levelAgent, levelAgent => new List<SingleAgentState>());
            var nextInitialStates = new Dictionary<char, SingleAgentState>(_level.AgentInitialStates);

            while (!IsAllMainGoalsSolved(solved))
            {
                var delegation = DelegateSubGoals(solved, nextInitialStates);
                foreach (var (key, value) in
                    CBSHandler.Search(_level.Agents,
                        delegation.NextInitialStates,
                        _level.Goals))
                {
                    if (value == null)
                    {
                        continue;
                    }

                    solutions[key].AddRange(value);
                    solved.AddRange(delegation.Goals[key]);
                    nextInitialStates[key] = value.Last();
                }
            }

            return solutions;
        }

        public Delegate DelegateSubGoals(List<MapItem> solved, Dictionary<char, SingleAgentState> nextInitialStates)
        {
            var subGoals = _level.Agents.ToDictionary(agent => agent, agent => new List<MapItem>());
            var agentInitialStates = new Dictionary<char, SingleAgentState>(nextInitialStates);

            foreach (var agent in _level.Agents)
            {
                var goalsToSolve = _level.Goals[agent].Where(goal => !solved.Contains(goal)).ToList();
                var boxGoals = goalsToSolve.Where(goal => char.IsLetter(goal.Value));

                if (!goalsToSolve.Any())
                {
                    continue;
                }

                MapItem selectedGoal;
                if (boxGoals.Any())
                {
                    selectedGoal = boxGoals.First();
                    var selectedBox = agentInitialStates[agent].Boxes.First();
                    foreach (var mapItem in agentInitialStates[agent].Boxes)
                    {
                        if (selectedBox.Position.Equals(mapItem.Position))
                        {
                            continue;
                        }

                        agentInitialStates[agent].Walls.Add($"{mapItem.Position.Row},{mapItem.Position.Column}");
                    }
                }
                else
                {
                    selectedGoal = goalsToSolve.First();
                }

                subGoals[agent].Add(selectedGoal);
            }

            return new Delegate
            {
                Goals = subGoals,
                NextInitialStates = agentInitialStates
            };
        }
    }
}