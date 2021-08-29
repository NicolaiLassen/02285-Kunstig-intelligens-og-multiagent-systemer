using System;
using System.Collections.Generic;
using System.Linq;
using MaMapF.Models;

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

        public static int MaxMoves = 7;

        public Dictionary<char, List<SingleAgentState>> Search()
        {
            var agents = _level.Agents;
            var solved = new List<MapItem>();
            var solutions = Enumerable.ToDictionary(agents, agent => agent, agent => new List<SingleAgentState>());
            var problems = Enumerable.ToDictionary(agents, agent => agent,
                agent => new SingleAgentProblem(_level.AgentInitialStates[agent]));

            while (!IsAllMainGoalsSolved(solved))
            {
                // Create sub problem for each agent
                foreach (var agent in agents)
                {
                    problems[agent].ResetMods();
                    var unsolvedAgentGoals =
                        Enumerable.ToList(Enumerable.Where(_level.Goals[agent], goal => !solved.Contains(goal)));
                    problems[agent] = CreateSubProblem(problems[agent], unsolvedAgentGoals, solved, problems);
                }

                var nextNode = CBSHandler.Search(problems);

                // If an agent could not finnish because it is blocked by a WallBox
                var wallBoxConstraint = nextNode.WallBoxConstraint;
                if (wallBoxConstraint != null)
                {
                    // Console.Error.WriteLine("wallBoxConstraint");
                    // Console.Error.WriteLine(wallBoxConstraint);
                    // Environment.Exit(0);
                    problems[wallBoxConstraint.Agent].Constraints.Add(wallBoxConstraint);
                    continue;
                }

                // problems.Values.ToList().ForEach(p => Console.Error.WriteLine(p));
                // Console.Error.WriteLine($"nextSolutions: {nextSolutions}");
                // Environment.Exit(0);

                Console.Error.WriteLine("AAAAAAAAAAAAAAAAAAA");


                var maxSolutionLength = Enumerable.Max(nextNode.Solutions.Values, s => s.Count);

                solved.Clear();
                foreach (var agent in agents)
                {
                    var goals = _level.Goals[agent];
                    var solution = nextNode.Solutions[agent];
                    var solutionLength = solution.Count;
                    var lastState = solution[solutionLength - 1];

                    // Check all goals
                    foreach (var goal in goals)
                    {
                        if (Enumerable.Any(lastState.AllMapItems, item => goal.Equals(item)))
                        {
                            solved.Add(goal);
                        }
                    }

                    solutions[agent] = solution;
                    problems[agent].InitialState = Enumerable.Last(solution);


                    Console.Error.WriteLine($"NODE.KEY: {agent}");
                    Console.Error.WriteLine($"NODE.VAL: {solution}");
                    solution.ForEach(s => Console.Error.WriteLine(s));
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


        private SingleAgentProblem CreateSubProblem(SingleAgentProblem previous, List<MapItem> unsolved,
            List<MapItem> solved, Dictionary<char, SingleAgentProblem> problems)
        {
            var agent = previous.InitialState.AgentName;
            var initialState = previous.InitialState;
            var problem = new SingleAgentProblem(previous.InitialState);
            var allBoxes = problem.InitialState.Boxes;

            // Sub goal: Remove blocking box for other agent
            if (Enumerable.Any(previous.Constraints))
            {
                problem.Type = SingleAgentProblemType.MoveBlock;
                problem.Constraints = previous.Constraints;

                // var blockPosition = problem.SelectedBox == null ? previous.Constraints.First().Position : previous.Constraints.Last().Position;
                var blockPosition = Enumerable.First(previous.Constraints).Position;
                problem.SelectedBox =
                    Enumerable.FirstOrDefault(initialState.Boxes, b => b.Position.Equals(blockPosition));

                // Select first block constraint and convert all other boxes to walls
                var nonBlockBoxes = Enumerable.Where(allBoxes, b => !blockPosition.Equals(b.Position));
                foreach (var box in nonBlockBoxes)
                {
                    problem.AddBoxMod(box);
                }

                return problem;
            }

            // Return problem with no goals if no unsolved goals left 
            if (!Enumerable.Any(unsolved))
            {
                // Convert all boxes to boxes to force blocking problem
                foreach (var box in allBoxes)
                {
                    problem.AddBoxMod(box);
                }

                return problem;
            }

            var unsolvedBoxGoals = Enumerable.ToList(Enumerable.Where(unsolved, goal => char.IsLetter(goal.Value)));

            // If no unsolved box goals then return agent problem
            // Sub goal: Move agent to agent goal position
            if (!Enumerable.Any(unsolvedBoxGoals))
            {
                problem.Type = SingleAgentProblemType.AgentToGoal;
                // Convert all boxes to walls to optimize a*
                foreach (var box in initialState.Boxes)
                {
                    problem.AddBoxMod(box);
                }

                problem.Goals.Add(Enumerable.First(unsolved));
                return problem;
            }

            var unusedBoxes = Enumerable.ToList(Enumerable.Where(allBoxes, box => !Enumerable.Any(solved, box.Equals)));

            // Sub goal: Move previously selected box to goal
            if (previous.Type == SingleAgentProblemType.AgentToBox)
            {
                // Add goal to problem
                problem.Type = SingleAgentProblemType.BoxToGoal;
                problem.Goals.Add(previous.SelectedBoxGoal);
                problem.SelectedBox = previous.SelectedBox;
                problem.SelectedBoxGoal = previous.SelectedBoxGoal;

                // Convert all non-selected boxes to walls
                var otherBoxes = Enumerable.Where(allBoxes, box => !previous.SelectedBox.Equals(box));
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
            var selectedBox = Enumerable.First(unusedBoxes);
            var selectedGoal = Enumerable.First(unsolvedBoxGoals);
            List<Position> neighbours = null;
            foreach (var goal in unsolvedBoxGoals)
            {
                foreach (var box in unusedBoxes)
                {
                    if (goal.Value != box.Value)
                    {
                        continue;
                    }

                    var boxNeighbours = Position.GetNeighbours(box.Position);
                    var freeNeighbours = new List<Position>();

                    foreach (var boxNeighbour in boxNeighbours)
                    {
                        var hasBlock = false;
                        foreach (var problemsKey in problems.Keys)
                        {
                            if (problems[problemsKey].InitialState.AllPositions.Contains(boxNeighbour))
                            {
                                hasBlock = true;
                            }
                        }

                        if (!hasBlock)
                        {
                            freeNeighbours.Add(boxNeighbour);
                        }
                    }

                    if (freeNeighbours.Count == 0)
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
                        neighbours = freeNeighbours;
                    }
                }
            }

            // Find best neighbour position to selected box
            if (neighbours == null)
            {
                Console.Error.WriteLine("JNWENWI");
                Environment.Exit(0);
            }

            // var otherAgentsBoxPositions = _level.AgentInitialStates.Values
            //     .Where(s => s.AgentName != agent)
            //     .SelectMany(s => s.Boxes)
            //     .Select(b => b.Position).ToList();

            var bestPosition = Enumerable.First(Enumerable.OrderBy(neighbours, p =>
            {
                var distance = Position.Distance(initialState.Agent.Position, p);
                // previous box goal bonus
                // other agent box penalty
                return distance;
            }));

            // Find the box with least neighbor block
            // Console.Error.WriteLine($"bestPosition: {bestPosition}");

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
                if (!Enumerable.All(goal, solved.Contains))
                {
                    allSolved = false;
                }
            }

            return allSolved;
        }
    }
}