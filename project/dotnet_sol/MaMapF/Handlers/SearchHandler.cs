using System;
using System.Collections.Generic;
using System.Linq;
using MaMapF.Models;
using Action = MaMapF.Models.Action;

//********************
// Try map A2 to see delegation in action
//*******************

// Remove waiting time for other agents to finnish their sub goals
// Improve search time for levels with corridor


namespace MaMapF.Handlers
{
    public class SearchHandler
    {
        public static bool Temp = true;

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
                    problems[agent].ResetMods();
                    var unsolvedAgentGoals = _level.Goals[agent].Where(goal => !solved.Contains(goal)).ToList();
                    problems[agent] = CreateSubProblem(problems[agent], unsolvedAgentGoals, solved);
                }

                var nextSolutions = CBSHandler.Search(problems);

                // If an agent could not finnish because it is blocked by a WallBox
                var wallBoxConstraint = nextSolutions.WallBoxConstraint;
                if (wallBoxConstraint != null)
                {
                    problems[wallBoxConstraint.Agent].Constraints.Add(wallBoxConstraint);
                    continue;
                }

                var maxSolutionLength = nextSolutions.Solutions.Values.Max(s => s.Count);
                foreach (var (agent, solution) in nextSolutions.Solutions)
                {
                    if (Temp && agent == '2')
                    {
                        Console.Error.WriteLine($"UUUUUUUUUUUUUUUUUUUUUUU");
                        Temp = false;
                        var sTemp = new List<SingleAgentState> {solution[0]};
                        for (int i = 0; i < maxSolutionLength; i++)
                        {
                            sTemp.Add(SingleAgentSearchHandler.CreateNextState(sTemp[i], Action.NoOp));
                        }
                    
                    
                        solutions[agent] = sTemp;
                        // solved.AddRange(problems[agent].Goals);
                        problems[agent].InitialState = sTemp.Last();
                        continue;
                    }

                    // Console.Error.WriteLine("AAAAAAAAAAAAAAAAAAA");
                    // problems[agent].Constraints.ForEach(s => Console.Error.WriteLine(s));
                    // solution.ForEach(s => Console.Error.WriteLine(s));


                    solutions[agent] = solution;
                    solved.AddRange(problems[agent].Goals);
                    problems[agent].InitialState = solution.Last();
                }
            }

            // foreach (var s in solutions.Values)
            // {
            //     Console.Error.WriteLine("---------------------------------");
            //     foreach (var state in s)
            //     {
            //         Console.Error.WriteLine(state);
            //     }
            // }

            return solutions;
        }

        private static bool IsNullSolution(List<SingleAgentState> solution)
        {
            var r = solution.Any(s => s.Action != null && s.Action != Action.NoOp);
            Console.Error.WriteLine(solution.First().AgentName);
            Console.Error.WriteLine(r);
            return r;
        }


        private SingleAgentProblem CreateSubProblem(SingleAgentProblem previous, List<MapItem> unsolved,
            List<MapItem> solved)
        {
            var agent = previous.InitialState.AgentName;
            var initialState = previous.InitialState;
            var problem = new SingleAgentProblem(previous.InitialState);
            var allBoxes = problem.InitialState.Boxes;


            // Sub goal: Remove blocking box for other agent
            if (previous.Constraints.Any())
            {
                problem.Type = SingleAgentProblemType.MoveBlock;
                problem.Constraints = previous.Constraints;

                // Select first block constraint and convert all other boxes to walls
                var blockPosition = previous.Constraints.First().Position;
                var nonBlockBoxes = allBoxes.Where(b => !blockPosition.Equals(b.Position));
                foreach (var box in nonBlockBoxes)
                {
                    problem.AddBoxMod(box);
                }

                return problem;
            }

            // Return problem with no goals if no unsolved goals left 
            if (!unsolved.Any())
            {
                // Convert all boxes to boxes to force blocking problem
                foreach (var box in allBoxes)
                {
                    problem.AddBoxMod(box);
                }

                return problem;
            }

            var unsolvedBoxGoals = unsolved.Where(goal => char.IsLetter(goal.Value)).ToList();

            // If no unsolved box goals then return agent problem
            // Sub goal: Move agent to agent goal position
            if (!unsolvedBoxGoals.Any())
            {
                problem.Type = SingleAgentProblemType.AgentToGoal;
                // Convert all boxes to walls to optimize a*
                foreach (var box in initialState.Boxes)
                {
                    problem.AddBoxMod(box);
                }

                problem.Goals.Add(unsolved.First());
                return problem;
            }

            var unusedBoxes = allBoxes.Where(box => !solved.Any(box.Equals)).ToList();

            // Sub goal: Move previously selected box to goal
            if (previous.Type == SingleAgentProblemType.AgentToBox)
            {
                // Add goal to problem
                problem.Type = SingleAgentProblemType.BoxToGoal;
                problem.Goals.Add(previous.SelectedBoxGoal);
                problem.SelectedBox = previous.SelectedBox;
                problem.SelectedBoxGoal = previous.SelectedBoxGoal;

                // Convert all non-selected boxes to walls
                var otherBoxes = allBoxes.Where(box => !previous.SelectedBox.Equals(box));
                foreach (var box in otherBoxes)
                {
                    problem.AddBoxMod(box);
                }

                
                Console.Error.WriteLine($"prevBox: {problem.SelectedBox}");
                return problem;
            }


            // Sub goal: Move agent to a box

            // Select "unused-box" and "unsolved-goal" with smallest distance
            // distance(agent, box) + distance(box, goal)
            var minDistance = Int32.MaxValue;
            var selectedBox = unusedBoxes.First();
            var selectedGoal = unsolvedBoxGoals.First();
            foreach (var goal in unsolvedBoxGoals)
            {
                foreach (var box in unusedBoxes)
                {
                    if (goal.Value != box.Value)
                    {
                        continue;
                    }
                    
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

            // Find best neighbour position to selected box
            var neighbours = Position.GetNeighbours(selectedBox.Position)
                .Where(p => !initialState.IsWall(p)).ToList();
            // var otherAgentsBoxPositions = _level.AgentInitialStates.Values
            //     .Where(s => s.AgentName != agent)
            //     .SelectMany(s => s.Boxes)
            //     .Select(b => b.Position).ToList();

            var bestPosition = neighbours.OrderBy(p =>
            {
                var distance = Position.Distance(initialState.Agent.Position, p);
                // previous box goal bonus
                // other agent box penalty
                return distance;
            }).First();

            Console.Error.WriteLine($"bestPosition: {bestPosition}");

            // Add agent position goal to problem
            problem.Type = SingleAgentProblemType.AgentToBox;
            problem.Goals.Add(new MapItem(agent, bestPosition));
            problem.SelectedBox = selectedBox;
            problem.SelectedBoxGoal = selectedGoal;

            // Convert all boxes to walls
            foreach (var box in allBoxes)
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