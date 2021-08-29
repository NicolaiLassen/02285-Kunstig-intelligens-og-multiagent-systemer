using System;
using System.Collections.Generic;
using System.Linq;
using MaMapF.Models;
using Priority_Queue;
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
        public static int COUNTER = 0;

        public SearchHandler(Level level)
        {
            _level = level;
        }

        // TODO FIND A WAY TO INCREMENT THIS IF THERE IS A BLOCKED AGENT
        // public static int MaxMoves = 100000000;

        public Dictionary<char, List<SingleAgentState>> Search()
        {
            var agents = _level.Agents;
            var goals = _level.Goals;
            var solved = agents.ToDictionary(agent => agent, agent => new List<MapItem>());
            var solutions = agents.ToDictionary(agent => agent, agent => new List<SingleAgentState>());
            var solvedAgents = new List<char>();
            var solvedSubGoal = new List<char>(_level.Agents);

            var problems = agents.ToDictionary(agent => agent,
                agent => new SingleAgentProblem(_level.AgentInitialStates[agent]));

            while (!IsAllAgentsDone(solved))
            {
                var solvedGoalsCount = solved.SelectMany(s => s.Value).Count();
                Console.Error.WriteLine($"{solvedGoalsCount}/{_level.GoalCount}");

                // Create sub problem for each agent
                foreach (var agent in agents)
                {
                    // Don't change the agent if its still on its path in life
                    if (!solvedSubGoal.Contains(agent)) continue;

                    problems[agent].ResetMods();
                    problems[agent] = CreateSubProblem(problems[agent], goals[agent], solved[agent], problems);

                    // Console.Error.WriteLine("problems[agent]:");
                    // Console.Error.WriteLine(problems[agent]);
                }

                solvedSubGoal = new List<char>();

                var nextNode = CBSHandler.Search(problems, solvedAgents);

                if (nextNode == null)
                {
                    Console.Error.WriteLine("WRONG FORMAT");
                    Environment.Exit(0);
                }


                // foreach (var key in nextNode.Solutions.Keys)
                // {
                //     if (nextNode.Solutions[key] == null)
                //     {
                //         nextNode.Solutions[key] = new List<SingleAgentState>() {problems[key].InitialState};
                //     }
                // }


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

                // var spinningAgents = solutions.Where(s => !solvedAgents.Contains(s.Key)).Count(s =>
                //     s.Value.Skip(1).Take(minUnsolvedSolutionLength - 1).All(s => s.Action.Type == ActionType.NoOp));

                // if (spinningAgents > 0 && minUnsolvedSolutionLength > 1)
                // {
                //     MaxMoves += 1;
                // }

                foreach (var agent in agents)
                {
                    var solution = nextNode.Solutions[agent];

                    if (solution.Count == minUnsolvedSolutionLength)
                    {
                        solvedSubGoal.Add(agent);
                    }

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


                // Console.Error.WriteLine($"MaxMoves: {MaxMoves}");
                // Console.Error.WriteLine($"minSolutionLength: {minUnsolvedSolutionLength}");
                // Console.Error.WriteLine($"maxSolutionLength: {maxSolutionLength}");
                // Console.Error.WriteLine($"solvedAgents: {solvedAgents.Count}");

                // solutions['1'].ForEach(s => Console.Error.WriteLine(s));

                //

                // TODO: KEEP IN MIND THAT WE HAVE A COUNTER BREAK!
                // if (COUNTER == 8)
                // {
                //     break;
                // }

                COUNTER += 1;
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
                // problem.Goals.Add(new MapItem(agent, initialState.Agent.Position));
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
            var orderedBoxGoals = new SimplePriorityQueue<Tuple<MapItem, MapItem>>();

            foreach (var goal in unsolvedBoxGoals)
            {
                foreach (var box in unusedBoxes)
                {
                    // Skip if goal does not match box
                    if (goal.Value != box.Value)
                    {
                        continue;
                    }

                    // Skip if box is already on goal position
                    if (goal.Position.Equals(box.Position))
                    {
                        continue;
                    }

                    // Skip if no path to target VIA BFS
                    // COULD BE USED TO COUNT CORRECT H
                    if (!BFSToPath(initialState, box.Position, initialState.Agent.Position))
                    {
                        continue;
                    }

                    var agentBoxDistance = Position.Distance(initialState.Agent, box);
                    var boxGoalDistance = Position.Distance(box, goal);
                    var distance = agentBoxDistance + boxGoalDistance;

                    orderedBoxGoals.Enqueue(new Tuple<MapItem, MapItem>(box, goal), distance);
                }
            }

            // TODO
            Tuple<MapItem, MapItem> selectedBoxGoal = null;
            foreach (var orderedBoxGoal in orderedBoxGoals)
            {
                if (!BFSToPath(initialState, orderedBoxGoal.Item1.Position, initialState.Agent.Position)) continue;
                selectedBoxGoal = orderedBoxGoal;
                break;
            }

            var selectedBox = selectedBoxGoal.Item1;
            var selectedGoal = selectedBoxGoal.Item2;


            // Find best neighbour position to selected box


            // var otherAgentsBoxPositions = _level.AgentInitialStates.Values
            //     .Where(s => s.AgentName != agent)
            //     .SelectMany(s => s.Boxes)
            //     .Select(b => b.Position).ToList();
            var neighbours = Position.GetNeighbours(selectedBox.Position);

            // JUST CHECK IF SPOT IS OPEN
            var neighboursReachable = neighbours.Where(n => !initialState.IsWall(n) && !initialState.IsBox(n));

            var bestPosition = neighboursReachable.OrderBy(p =>
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

        public bool BFSToPath(SingleAgentState state, Position start, Position end)
        {
            Queue<Position> struc = new Queue<Position>();
            var visited = new HashSet<Position>();
            struc.Enqueue(start);

            while (struc.Count > 0)
            {
                var p = struc.Dequeue();

                if (visited.Contains(p))
                {
                    continue;
                }

                if (p.Equals(end))
                {
                    return true;
                }

                visited.Add(p);

                var neighbours = Position.GetNeighbours(p);
                foreach (var neighbour in neighbours)
                {
                    if (state.IsWall(neighbour))
                    {
                        continue;
                    }

                    if (state.IsBox(neighbour))
                    {
                        continue;
                    }

                    struc.Enqueue(neighbour);
                }
            }

            return false;
        }

        private bool IsAllAgentsDone(Dictionary<char, List<MapItem>> solved)
        {
            return _level.GoalCount == solved.Values.Sum(s => s.Count);
        }

        private bool IsAgentDone(char agent, List<MapItem> solved)
        {
            var goals = _level.Goals[agent];
            return goals.All(g => solved.Any(g.Equals));
        }
    }
}