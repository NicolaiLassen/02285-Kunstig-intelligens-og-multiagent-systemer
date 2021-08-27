using System;
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
            var solutions =
                _level.Agents.ToDictionary(levelAgent => levelAgent, levelAgent => new List<SingleAgentState>());

            var goals = new Dictionary<char, List<MapItem>>(_level.Goals);
            var agentInitialStates = new Dictionary<char, SingleAgentState>(_level.AgentInitialStates);
            var solved = new List<MapItem>();

            while (!IsAllMainGoalsSolved(solved))
            {
                var subGoals = new Dictionary<char, List<MapItem>>();
                var subGoalInitialStates = new Dictionary<char, List<List<char>>>();
                foreach (var agent in _level.Agents)
                {
                    var goalsToSolve = goals[agent].Where(goal => !solved.Contains(goal)).ToList();
                    var boxGoals = goalsToSolve.Where(goal => char.IsLetter(goal.Value));

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

                    if (selectedGoal == null)
                    {
                        continue;
                    }
                    
                    subGoals.Add(agent, new List<MapItem> {selectedGoal});
                }

                foreach (var (key, value) in CBSHandler.Search(subGoals.Keys.ToList(), agentInitialStates, subGoals))
                {
                    if (value == null)
                    {
                        continue;
                    }

                    solutions[key].AddRange(value);
                    solved.AddRange(subGoals[key]);
                    agentInitialStates[key] = value.Last();
                }
            }

            return solutions;
        }
    }
}