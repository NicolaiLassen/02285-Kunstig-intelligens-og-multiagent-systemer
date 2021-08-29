using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MaMapF.Models;

namespace MaMapF.Handlers
{
    public class ServerHandler
    {
        private const string CLIENT_NAME = "client46";

        public static Level GetServerLevel()
        {
            var serverLines = GetServerOut();
            var colorsIndex = serverLines.IndexOf("#colors");
            var initialIndex = serverLines.IndexOf("#initial");
            var goalIndex = serverLines.IndexOf("#goal");
            var endIndex = serverLines.IndexOf("#end");

            var colors = new Dictionary<char, string>();
            var colorRangeCount = (initialIndex - colorsIndex) - 1;
            foreach (var line in serverLines.GetRange(colorsIndex + 1, colorRangeCount))
            {
                var lineSplit = line.Split(":");
                var color = lineSplit[0].Trim().ToLower();
                var items = lineSplit[1].Split(",").Select(s => s.Trim());
                foreach (var item in items)
                {
                    colors.Add(Convert.ToChar(item), color);
                }
            }

            var agents = colors.Keys.Where(char.IsDigit).Select(agent => char.Parse($"{agent}")).ToList();

            var initialRangeCount = (goalIndex - initialIndex) - 1;
            var initialLines = LinesToCharMatrix(serverLines.GetRange(initialIndex + 1, initialRangeCount));

            //
            // for (var row = 0; row < initialLines.Count; row++)
            // {
            //     for (var col = 0; col < initialLines[row].Count; col++)
            //     {
            //         var c = initialLines[row][col];
            //         if (char.IsLetter(c))
            //         {
            //             if (agents.All(a => colors[a] != colors[c]))
            //             {
            //                 initialLines[row][col] = '+';
            //             }
            //         }
            //     }
            // }


            var goalRangeCount = (endIndex - goalIndex) - 1;
            var goalLines = LinesToCharMatrix(serverLines.GetRange(goalIndex + 1, goalRangeCount));

            var goals = new Dictionary<char, List<MapItem>>();

            foreach (var agent in agents)
            {
                goals.Add(agent, new List<MapItem>());
                for (var row = 0; row < goalLines.Count; row++)
                {
                    for (var col = 0; col < goalLines[row].Count; col++)
                    {
                        var c = goalLines[row][col];
                        if (!char.IsLetter(c) && c != agent) continue;
                        if (colors[agent] != colors[c]) continue;
                        goals[agent].Add(
                            new MapItem(c, new Position(row, col))
                        );
                    }
                }
            }

            return new Level(colors, agents, initialLines, goalLines, goals);
        }

        public static void SendServerPlan(Dictionary<char, List<SingleAgentState>> plan)
        {
            var maxLength = plan.Values.Max(p => p.Count);
            for (var step = 1; step < maxLength; step++)
            {
                var command = string.Join("|", plan.Keys.Select(agent => plan[agent][step].Action));
                Console.WriteLine(command);
            }
        }

        private static List<string> GetServerOut()
        {
            // Set OpenStandardInput
            Console.SetIn(new StreamReader(Console.OpenStandardInput()));
            Console.WriteLine(CLIENT_NAME);
            Console.Out.Flush();

            var serverLines = new List<string>();
            var line = "";

            while (!line.StartsWith("#end"))
            {
                line = Console.ReadLine();
                line = line.StartsWith("#") ? line.Trim().Replace("\n", "") : line.Replace("\n", "");
                serverLines.Add(line);
            }

            return serverLines;
        }

        private static List<List<char>> LinesToCharMatrix(IEnumerable<string> lines) =>
            lines.Select(line => line.ToCharArray().ToList()).ToList();
    }
}