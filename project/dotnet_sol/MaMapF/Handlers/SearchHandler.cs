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

                problem.Goals.Add(unsolved.First());
                return problem;
            }


            // TODO only boxes not used for other goals
            var allBoxes = problem.InitialState.Boxes;
            var unusedBoxes = allBoxes.Where(box => !solved.Any(box.Equals)).ToList();


            // Select "unused-box" and "unsolved-goal" with smallest distance
            // distance(agent, box) + distance(box, goal)
            var minDistance = Int32.MaxValue;
            var selectedBox = unusedBoxes.First();
            var selectedGoal = unsolvedBoxGoals.First();
            foreach (var goal in unsolvedBoxGoals)
            {
                foreach (var box in unusedBoxes)
                {
                    var agentBoxDistance = Position.Distance(initialState.Agent, box);
                    var boxGoalDistance = Position.Distance(box, goal);
                    var distance = agentBoxDistance + boxGoalDistance;
                    if (distance < minDistance)
                    {
                        minDistance = distance;
                        selectedBox = box;
                        selectedGoal = goal;
                    }
                }
            }

            // Add goal to problem
            problem.Goals.Add(selectedGoal);

            // Convert all other boxes to walls
            var otherBoxes = allBoxes.Where(box => !selectedBox.Equals(box));
            foreach (var box in otherBoxes)
            {
                problem.AddBoxMod(box);
            }

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