using System;
using System.Collections.Generic;
using System.Linq;
using MaMapF.Models;

namespace MaMapF.Handlers
{
    public class SearchHandler
    {
        private readonly Level _level;
        private List<MapItem> Solved { get; set; }

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
            var solutions =
                _level.Agents.ToDictionary(levelAgent => levelAgent, levelAgent => new List<SingleAgentState>());
            
            var agentInitialStates = new Dictionary<char, SingleAgentState>(_level.AgentInitialStates);

            while (!IsAllMainGoalsSolved())
            {
                var delegation = DelegateSubGoals();
                foreach (var (key, value) in CBSHandler.Search(_level.Agents, delegation.NextInitialStates,
                    delegation.Goals))
                {
                    if (value == null)
                    {
                        continue;
                    }

                    solutions[key].AddRange(value);
                    solved.AddRange(delegation.Goals[key]);
                    agentInitialStates[key] = value.Last();
                }
            }

            return solutions;
        }

        public GoalDelegate DelegateSubGoals()
        {
            var goals = new Dictionary<char, List<MapItem>>(_level.Goals);
            var subGoals = _level.Agents.ToDictionary(agent => agent, agent => new List<MapItem>());
            var subGoalInitialStates = new Dictionary<char, List<List<char>>>();

            foreach (var agent in _level.Agents)
            {
                var goalsToSolve = goals[agent].Where(goal => !solved.Contains(goal)).ToList();
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
                        Console.Error.WriteLine(selectedGoal);
                        Console.Error.WriteLine(mapItem);
                        if (selectedBox.Position.Equals(mapItem.Position))
                        {
                            continue;
                        }

                        agentInitialStates[agent].Map[mapItem.Position.Row][mapItem.Position.Column] = '+';
                    }
                }
                else
                {
                    selectedGoal = goalsToSolve.First();
                }

                subGoals[agent].Add(selectedGoal);
            }

            return new GoalDelegate();
        }
    }
}