using System;
using System.Collections.Generic;
using System.Linq;
using MaMapF.Models;
using Priority_Queue;
using Action = MaMapF.Models.Action;

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

        public Dictionary<char, List<SingleAgentState>> Search()
        {
            var agents = _level.Agents;
            var goals = _level.Goals;
            var solved =
                agents.ToDictionary(agent => agent, agent => new List<MapItem>());
            var solutions =
                agents.ToDictionary(agent => agent, agent => new List<SingleAgentState>());

            var solvedAgents = new List<char>();
            var agentsToDelegate = new List<char>(_level.Agents);
            var pastSolutionLength = 1;

            var problems = agents.ToDictionary(agent => agent,
                agent => new SingleAgentProblem(_level.AgentInitialStates[agent]));

            while (!IsAllAgentsDone(solved))
            {
                // Create sub problem for each agent
                foreach (var agent in agents)
                {
                    // Don't change the agent if its still on its path in life
                    if (!agentsToDelegate.Contains(agent)) continue;

                    problems[agent].ResetMods();
                    problems[agent] = CreateSubProblem(problems[agent], goals[agent], solved[agent], problems);
                }

                // Reset for next round trip
                agentsToDelegate = new List<char>();

                var nextNode = CBSHandler.Search(problems, solvedAgents, pastSolutionLength);
                if (nextNode == null)
                {
                    Console.Error.WriteLine("!NO SOLUTION FOUND!");
                    Environment.Exit(0);
                }

                var minUnsolvedSolutionLength = nextNode.Solutions
                    .Where(k => !solvedAgents.Contains(k.Key))
                    .Min(s => s.Value.Count);

                // minUnsolvedSolutionLength =
                //     Math.Min(minUnsolvedSolutionLength, pastSolutionLength + cbsHorizonMinimum);

                foreach (var agent in agents)
                {
                    var solution = nextNode.Solutions[agent];

                    if (solution.Count == minUnsolvedSolutionLength)
                    {
                        agentsToDelegate.Add(agent);
                    }

                    // If I'm the guy or I'm still going then cut me off
                    if (solution.Count >= minUnsolvedSolutionLength)
                    {
                        solutions[agent] = solution.GetRange(0, minUnsolvedSolutionLength);
                        problems[agent].InitialState = solutions[agent].Last();
                        continue;
                    }

                    // I'm done then fill me up with NoOps
                    var solutionDiff = Math.Abs(solution.Count - minUnsolvedSolutionLength);
                    solutions[agent] = solution.GetRange(0, solution.Count);
                    var nextState = solutions[agent].Last();
                    for (int i = 0; i < solutionDiff; i++)
                    {
                        nextState = SingleAgentSearchHandler.CreateNextState(nextState, Action.NoOp);
                        solutions[agent].Add(nextState);
                    }

                    problems[agent].InitialState = solutions[agent].Last();
                }

                solved = agents.ToDictionary(agent => agent, agent =>
                {
                    var solution = solutions[agent];
                    var lastState = solution.Last();
                    return goals[agent].Where(g => lastState.AllMapItems.Any(g.Equals)).ToList();
                });

                solvedAgents = agents.Where(a => IsAgentDone(a, solved[a])).ToList();
                pastSolutionLength = minUnsolvedSolutionLength;
                
                // if (COUNTER == 0)
                // {
                //     break;
                // }
                //
                // COUNTER += 1;
            }

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

            // Return problem with no goals if no unsolved goals left 
            if (!unsolved.Any())
            {
                foreach (var box in initialState.Boxes)
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

                return problem;
            }

            // Sub goal: Move agent to a box
            // Select "unused-box" and "unsolved-goal" with smallest distance
            // distance(agent, box) + distance(box, goal)
            var orderedBoxGoals = new SimplePriorityQueue<BoxGoal>();

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

                    var boxGoalDistance = Position.Distance(box, goal);
                    orderedBoxGoals.Enqueue(new BoxGoal {Box = box, Goal = goal}, boxGoalDistance);
                }
            }

            var boxGoal =
                // ReSharper disable once PossibleNullReferenceException
                // SHOULD NOT BE NULL
                orderedBoxGoals.FirstOrDefault(orderedBoxGoal =>
                    IsReachableBest(initialState,
                        orderedBoxGoal.Box.Position,
                        initialState.Agent.Position));

            // JUST SELECT CLOSEST POS THEN
            if (boxGoal == null)
            {
                // Convert all boxes to walls
                foreach (var box in allBoxes)
                {
                    problem.AddBoxMod(box);
                }
                return problem;
            }
            
            // Find best neighbour position to selected box
            var neighbours = Position.GetNeighbours(boxGoal.Box.Position);

            // JUST CHECK IF SPOT IS OPEN
            var neighboursReachable =
                neighbours.Where(n =>
                    !initialState.IsWall(n) &&
                    !initialState.IsBox(n));
            
            var bestPosition = neighboursReachable.OrderBy(p =>
            {
                var distance = Position.Distance(boxGoal.Goal.Position, p);
                // previous box goal bonus
                // other agent box penalty
                return distance;
            }).LastOrDefault();
            
            // Add agent position goal to problem
            problem.Type = SingleAgentProblemType.AgentToBox;
            problem.Goals.Add(new MapItem(agent, bestPosition));
            problem.SelectedBox = boxGoal.Box;
            problem.SelectedBoxGoal = boxGoal.Goal;

            // Convert all boxes to walls
            foreach (var box in allBoxes)
            {
                problem.AddBoxMod(box);
            }

            return problem;
        }

        private bool IsReachableBest(SingleAgentState state, Position start, Position end)
        {
            var queue = new SimplePriorityQueue<Position>();
            var visited = new HashSet<Position>();
            queue.Enqueue(start, 0);

            while (queue.Count > 0)
            {
                var p = queue.Dequeue();

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
                foreach (var neighbour in
                    neighbours.Where(neighbour => !state.IsWall(neighbour))
                        .Where(neighbour => !state.IsBox(neighbour)))
                {
                    queue.Enqueue(neighbour, Position.Distance(neighbour, end));
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