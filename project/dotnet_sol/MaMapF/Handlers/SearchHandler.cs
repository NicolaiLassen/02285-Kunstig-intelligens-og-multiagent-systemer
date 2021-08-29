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
        private readonly Level _level;
        public int COUNTER = 0;

        public SearchHandler(Level level)
        {
            _level = level;
        }

        // TODO FIND A WAY TO INCREMENT THIS IF THERE IS A BLOCKED AGENT
        public static int MaxMoves = 2;

        public Dictionary<char, List<SingleAgentState>> Search()
        {
            var agents = _level.Agents;
            var goals = _level.Goals;
            var solved = agents.ToDictionary(agent => agent, agent => new List<MapItem>());
            var solutions = agents.ToDictionary(agent => agent, agent => new List<SingleAgentState>());
            var solvedAgents = new List<char>();


            var problems = agents.ToDictionary(agent => agent,
                agent => new SingleAgentProblem(_level.AgentInitialStates[agent]));

            while (!IsAllAgentsDone(solved))
            {
                // Create sub problem for each agent
                foreach (var agent in agents)
                {
                    problems[agent].ResetMods();
                    problems[agent] = CreateSubProblem(problems[agent], goals[agent], solved[agent], problems);

                    // Console.Error.WriteLine("problems[agent]:");
                    // Console.Error.WriteLine(problems[agent]);
                }

                var nextNode = CBSHandler.Search(problems);


                // If an agent could not finnish because it is blocked by a WallBox
                // var wallBoxConstraint = nextNode.WallBoxConstraint;
                // if (wallBoxConstraint != null)
                // {
                //     problems[wallBoxConstraint.Agent].Constraints.Add(wallBoxConstraint);
                //     continue;
                // }


                var maxSolutionLength = nextNode.Solutions.Values.Max(s => s.Count);
                var minUnsolvedSolutionLength = nextNode.Solutions
                    .Where(k => !solvedAgents.Contains(k.Key))
                    .Min(s => s.Value.Count);

                var spinningAgents = solutions.Where(s => !solvedAgents.Contains(s.Key)).Count(s =>
                    s.Value.Skip(1).Take(minUnsolvedSolutionLength - 1).All(s => s.Action.Type == ActionType.NoOp));

                // if (spinningAgents > 0 && minUnsolvedSolutionLength > 1)
                // {
                //     MaxMoves += 1;
                // }

                if (maxSolutionLength == MaxMoves)
                {
                    MaxMoves = 2;
                }
                else if (maxSolutionLength > MaxMoves)
                {
                    MaxMoves += 1;
                    continue;
                }


                foreach (var agent in agents)
                {
                    var solution = nextNode.Solutions[agent];

                    // If I'm the guy or I'm still going then cut me off
                    if (solution.Count >= minUnsolvedSolutionLength)
                    {
                        solutions[agent] = solution.GetRange(0, minUnsolvedSolutionLength);
                        problems[agent].InitialState = solutions[agent].Last();
                        continue;
                    }

                    // I'm done then fill me up
                    var solutionDiff = Math.Abs(solution.Count - minUnsolvedSolutionLength);
                    solutions[agent] = solution.GetRange(0, solution.Count);
                    var nextState = solutions[agent].Last();
                    for (int i = 0; i < solutionDiff; i++)
                    {
                        nextState = SingleAgentSearchHandler.CreateNextState(nextState, Action.NoOp);
                        solutions[agent].Add(nextState);
                    }

                    problems[agent].InitialState = solutions[agent].Last();


                    // Console.Error.WriteLine($"Solution.count: {solution.Count}");
                    // Console.Error.WriteLine($"minSolutionLength - 1: {minSolutionLength - 1}");

                    // solution.ForEach(s => Console.Error.WriteLine(s));


                    // Console.Error.WriteLine($"NODE.KEY: {agent}");
                    // Console.Error.WriteLine($"NODE.VAL: {solution}");
                    // solution.ForEach(s => Console.Error.WriteLine(s));
                }

                solved = agents.ToDictionary(agent => agent, agent =>
                {
                    var solution = solutions[agent];
                    var lastState = solution.Last();
                    return goals[agent].Where(g => lastState.AllMapItems.Any(g.Equals)).ToList();
                });
                solvedAgents = agents.Where(a => IsAgentDone(a, solved[a])).ToList();



                Console.Error.WriteLine($"MaxMoves: {MaxMoves}");
                Console.Error.WriteLine($"minSolutionLength: {minUnsolvedSolutionLength}");
                Console.Error.WriteLine($"maxSolutionLength: {maxSolutionLength}");
                Console.Error.WriteLine($"unsolvedAgents: {solvedAgents.Count}");

                // solutions['0'].ForEach(s => Console.Error.WriteLine(s));

                //
                // if (COUNTER == 1)
                // {
                //     break;
                // }
                //
                // COUNTER += 1;
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


        private SingleAgentProblem CreateSubProblem(SingleAgentProblem previous, List<MapItem> goals,
            List<MapItem> solved, Dictionary<char, SingleAgentProblem> problems)
        {
            var unsolved = goals.Where(goal => !solved.Contains(goal)).ToList();


            var agent = previous.InitialState.AgentName;
            var initialState = previous.InitialState;
            var problem = new SingleAgentProblem(previous.InitialState);
            var allBoxes = problem.InitialState.Boxes;

            // Sub goal: Remove blocking box for other agent
            // if (previous.Constraints.Any())
            // {
            //     problem.Type = SingleAgentProblemType.MoveBlock;
            //     problem.Constraints = previous.Constraints;
            //
            //     // // var blockPosition = problem.SelectedBox == null ? previous.Constraints.First().Position : previous.Constraints.Last().Position;
            //     // var blockPosition = previous.Constraints.First().Position;
            //     // problem.SelectedBox = initialState.Boxes.FirstOrDefault(b => b.Position.Equals(blockPosition));
            //     //
            //     // // Select first block constraint and convert all other boxes to walls
            //     // var nonBlockBoxes = allBoxes.Where(b => !blockPosition.Equals(b.Position));
            //     // foreach (var box in nonBlockBoxes)
            //     // {
            //     //     problem.AddBoxMod(box);
            //     // }
            //
            //
            //     return problem;
            // }

            // Return problem with no goals if no unsolved goals left 
            if (!unsolved.Any())
            {
                // Convert all boxes to boxes to force blocking problem
                // foreach (var box in allBoxes)
                // {
                //     problem.AddBoxMod(box);
                // }

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


                // Console.Error.WriteLine($"prevBox: {problem.SelectedBox}");
                return problem;
            }


            // Sub goal: Move agent to a box

            // Select "unused-box" and "unsolved-goal" with smallest distance
            // distance(agent, box) + distance(box, goal)
            var minDistance = Int32.MaxValue;
            var selectedBox = unusedBoxes.First();
            var selectedGoal = unsolvedBoxGoals.First();
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

                    // TODO MAKE PRIORITY

                    foreach (var boxNeighbour in boxNeighbours)
                    {
                        var hasBlock = false;
                        foreach (var otherAgent in problems.Keys)
                        {
                            if (problems[otherAgent].InitialState.IsWall(boxNeighbour))
                            {
                                hasBlock = true;
                            }


                            if (problems[otherAgent].InitialState.Boxes.Select(b => b.Position).Contains(boxNeighbour))
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

            var bestPosition = neighbours.OrderBy(p =>
            {
                var distance = Position.Distance(initialState.Agent.Position, p);
                // previous box goal bonus
                // other agent box penalty
                return distance;
            }).First();

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

        private bool IsAllAgentsDone(Dictionary<char, List<MapItem>> solved)
        {
            return _level.Agents.All(agent => IsAgentDone(agent, solved[agent]));
        }

        private bool IsAgentDone(char agent, List<MapItem> solved)
        {
            var goals = _level.Goals[agent];
            return goals.All(g => solved.Any(g.Equals));
        }
    }
}