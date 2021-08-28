using System;
using System.Collections.Generic;
using System.Linq;
using MaMapF.Models;
using Delegate = MaMapF.Models.Delegate;

//********************
// Try map A2 to see delegation in action
//*******************

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
            var solutions = _level.Agents.ToDictionary(agent => agent, agent => new List<SingleAgentState>());
            var nextInitialStates = new Dictionary<char, SingleAgentState>(_level.AgentInitialStates);
            var delegation = new Delegate
            {
                InitialStates = nextInitialStates
            };

            while (!IsAllMainGoalsSolved(solved))
            {
                DecorateSubGoals(delegation, solved);


                var nextSolutions = CBSHandler.Search(_level.Agents, delegation.InitialStates, delegation.Goals);
                foreach (var (agent, solution) in nextSolutions)
                {
                    if (solution == null)
                    {
                        // TODO change delegation (delete some box-walls etc) and trie again
                        continue;
                    }

                    solutions[agent] = solution;
                    solved.AddRange(delegation.Goals[agent]);
                    delegation.InitialStates[agent] = solution.Last();
                    delegation.ResetInitialStates();
                }
            }

            return solutions;
        }

        public void DecorateSubGoals(Delegate delegation, List<MapItem> solved)
        {
            var subGoals = _level.Agents.ToDictionary(agent => agent, agent => new List<MapItem>());
            var wallModifications = _level.Agents.ToDictionary(agent => agent, agent => new List<Position>());
            var boxModifications = _level.Agents.ToDictionary(agent => agent, agent => new List<MapItem>());
            var agentInitialStates = new Dictionary<char, SingleAgentState>(delegation.InitialStates);

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
                    // TODO: SELECT BETTER

                    selectedGoal = boxGoals.First();
                    // COULD BE MIN DONO
                    var maxBoxGoalDistance = 0;

                    // Solve the goals that is fathest from our agent
                    foreach (var boxGoal in boxGoals)
                    {
                        var distance = Position.Distance(agentInitialStates[agent].Agent, boxGoal);
                        if (distance <= maxBoxGoalDistance) continue;

                        maxBoxGoalDistance = distance;
                        selectedGoal = boxGoal;
                    }

                    MapItem selectedBox = agentInitialStates[agent].Boxes.First();
                    var minBoxDistance = Int32.MaxValue;

                    // pick the box that is closest to our goal
                    foreach (var box in agentInitialStates[agent].Boxes)
                    {
                        if (solved.Any(s => s.Position.Equals(box.Position)))
                        {
                            continue;
                        }

                        var distance = Position.Distance(selectedGoal, box);
                        if (distance >= minBoxDistance) continue;

                        // check if the box can be moved to the goal "block"!
                        minBoxDistance = distance;
                        selectedBox = box;
                    }

                    // Remove boxes and add walls
                    foreach (var box in agentInitialStates[agent].Boxes)
                    {
                        if (selectedBox.Position.Equals(box.Position))
                        {
                            continue;
                        }

                        // Modify the map to optimize for a*
                        wallModifications[agent].Add(box.Position);
                        boxModifications[agent].Add(box);
                        agentInitialStates[agent].Walls.Add($"{box.Position.Row},{box.Position.Column}");
                    }

                    delegation.UsedBoxes.Add(selectedBox.UID);
                }
                else
                {
                    // If all boxes are in place solve the agent problem
                    // TODO: this should still make walls for all boxes
                    // be aware of block
                    // TODO make func CLEAN THIS UP
                    foreach (var box in agentInitialStates[agent].Boxes)
                    {
                        // Modify the map to optimize for a*
                        wallModifications[agent].Add(box.Position);
                        boxModifications[agent].Add(box);
                        agentInitialStates[agent].Walls.Add($"{box.Position.Row},{box.Position.Column}");
                    }

                    selectedGoal = goalsToSolve.First();
                }

                Console.Error.WriteLine(selectedGoal);
                // Delegate the task to the agent
                subGoals[agent].Add(selectedGoal);
                // Remove boxes
                agentInitialStates[agent].Boxes =
                    agentInitialStates[agent].Boxes.Except(boxModifications[agent]).ToList();
            }

            delegation.Goals = subGoals;
            delegation.WallModifications = wallModifications;
            delegation.BoxModifications = boxModifications;
            delegation.InitialStates = agentInitialStates;
        }
    }
    
    
    
    public class SingleAgentProblem
    {
        public char AgentName { get; set; }
        public SingleAgentState InitialState { get; set; }
        public List<MapItem> Goals { get; set; } = new List<MapItem>();

        public List<Position> WallModifications { get; set; } = new List<Position>();
        public List<MapItem> BoxModifications { get; set; } = new List<MapItem>();

        public void ResetInitialState()
        {
            foreach (var position in WallModifications)
            {
                InitialState.Walls.Remove($"{position.Row},{position.Column}");
            }

            foreach (var box in BoxModifications)
            {
                InitialState.Boxes.Add(box);
            }

            WallModifications = new List<Position>();
            BoxModifications = new List<MapItem>();
        }

        public override string ToString()
        {
            var goalString = string.Join("\n", Goals.Select(g => g.ToString()));
            return $"SingleAgentProblem {AgentName}\n" +
                   $"{InitialState}" +
                   $"GOALS\n{goalString}\n";
        }
    }
    
}