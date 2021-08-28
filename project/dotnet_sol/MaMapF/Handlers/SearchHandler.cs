using System;
using System.Collections.Generic;
using System.Linq;
using MaMapF.Models;

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

        public Dictionary<char, List<SingleAgentState>> Search()
        {
            var agents = _level.Agents;
            var solved = new List<MapItem>();
            var solutions = agents.ToDictionary(agent => agent, agent => new List<SingleAgentState>());


            var problems = agents.ToDictionary(agent => agent, agent => new SingleAgentProblem
            {
                AgentName = agent,
                InitialState = _level.AgentInitialStates[agent],
            });


            while (!IsAllMainGoalsSolved(solved))
            {
                CreateSubProblems(problems, solved);


                var nextSolutions = CBSHandler.Search(problems);

                foreach (var (agent, solution) in nextSolutions)
                {
                    if (solution == null)
                    {
                        // TODO change delegation (delete some box-walls etc) and trie again
                        continue;
                    }

                    solutions[agent] = solution;
                    solved.AddRange(problems[agent].Goals);
                    problems[agent].InitialState = solution.Last();
                    problems[agent].Reset();
                }
            }

            return solutions;
        }

        public void CreateSubProblems(Dictionary<char, SingleAgentProblem> problems,
            List<MapItem> solved)
        {
            var agents = problems.Keys;

            foreach (var agent in agents)
            {
                var problem = problems[agent];
                var initialState = problem.InitialState;
                var goals = new List<MapItem>();
                var boxMods = new List<MapItem>();
                var wallMods = new List<Position>();


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
                        var distance = Position.Distance(initialState.Agent, boxGoal);
                        if (distance <= maxBoxGoalDistance) continue;

                        maxBoxGoalDistance = distance;
                        selectedGoal = boxGoal;
                    }

                    MapItem selectedBox = initialState.Boxes.First();
                    var minBoxDistance = Int32.MaxValue;

                    // pick the box that is closest to our goal
                    foreach (var box in initialState.Boxes)
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
                    foreach (var box in initialState.Boxes)
                    {
                        if (selectedBox.Position.Equals(box.Position))
                        {
                            continue;
                        }

                        // Modify the map to optimize for a*
                        wallMods.Add(box.Position);
                        boxMods.Add(box);
                        initialState.Walls.Add($"{box.Position.Row},{box.Position.Column}");
                    }

                    // TODO used boxes
                    // delegation.UsedBoxes.Add(selectedBox.UID);
                }
                else
                {
                    // If all boxes are in place solve the agent problem
                    // TODO: this should still make walls for all boxes
                    // be aware of block
                    // TODO make func CLEAN THIS UP
                    foreach (var box in initialState.Boxes)
                    {
                        // Modify the map to optimize for a*
                        wallMods.Add(box.Position);
                        boxMods.Add(box);
                        initialState.Walls.Add($"{box.Position.Row},{box.Position.Column}");
                    }

                    selectedGoal = goalsToSolve.First();
                }

                // Console.Error.WriteLine(selectedGoal);
                // Delegate the task to the agent
                goals.Add(selectedGoal);
                // Remove boxes
                initialState.Boxes = initialState.Boxes.Except(boxMods).ToList();


                problem.Goals = goals;
                problem.BoxModifications = boxMods;
                problem.WallModifications = wallMods;
            }
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
    }
}