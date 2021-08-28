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


            var problems = agents.ToDictionary(agent => agent,
                agent => new SingleAgentProblem(_level.AgentInitialStates[agent]));


            while (!IsAllMainGoalsSolved(solved))
            {
                // Create sub problem for each agent
                foreach (var agent in agents)
                {
                    var unsolvedAgentGoals = _level.Goals[agent].Where(goal => !solved.Contains(goal)).ToList();
                    problems[agent] = CreateSubProblem(problems[agent].InitialState, unsolvedAgentGoals, solved);
                }


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


        private static SingleAgentProblem CreateSubProblem(SingleAgentState initialState, List<MapItem> unsolved,
            List<MapItem> solved)
        {
            var problem = new SingleAgentProblem(initialState);

            // Return problem with no goals if no unsolved goals left 
            if (!unsolved.Any())
            {
                return problem;
            }


            var unsolvedBoxGoals = unsolved.Where(goal => char.IsLetter(goal.Value)).ToList();

            // If no unsolved box goals then return agent problem
            if (!unsolvedBoxGoals.Any())
            {
                // Convert all boxes to walls to optimize a*
                foreach (var box in initialState.Boxes)
                {
                    problem.AddBoxMod(box);
                }

                initialState.Boxes = initialState.Boxes.Except(problem.BoxMods).ToList();
                problem.Goals.Add(unsolved.First());
                return problem;
            }


            MapItem selectedGoal;
            // TODO: SELECT BETTER

            selectedGoal = unsolvedBoxGoals.First();
            // COULD BE MIN DONO
            var maxBoxGoalDistance = 0;

            // Solve the goals that is fathest from our agent
            foreach (var boxGoal in unsolvedBoxGoals)
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
                problem.WallMods.Add(box.Position);
                problem.BoxMods.Add(box);
                initialState.Walls.Add($"{box.Position.Row},{box.Position.Column}");
            }

            // TODO used boxes
            // delegation.UsedBoxes.Add(selectedBox.UID);


            // Console.Error.WriteLine(selectedGoal);
            // Delegate the task to the agent
            problem.Goals.Add(selectedGoal);
            // Remove boxes
            initialState.Boxes = initialState.Boxes.Except(problem.BoxMods).ToList();

            return problem;
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