using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MaMapF.Models;

namespace MaMapF
{
    public class ParserHandler
    {
        private const string CLIENT_NAME = "client46";

        public Level GetServerLevel()
        {
            var serverLines = GetServerOut();
            var colorsIndex = serverLines.IndexOf("#colors");
            var initialIndex = serverLines.IndexOf("#initial");
            var goalIndex = serverLines.IndexOf("#goal");
            var endIndex = serverLines.IndexOf("#end");

            var colors = new Dictionary<char, char>();
            var colorRangeCount = (initialIndex - colorsIndex) - 1;
            foreach (var line in serverLines.GetRange(colorsIndex + 1, colorRangeCount))
            {
                var lineSplit = line.Split(":");
                var color = Convert.ToChar(lineSplit[0].Trim().ToLower());
                var items = lineSplit[1].Split(",").Select(s => s.Trim());
                foreach (var item in items)
                {
                    colors.Add(Convert.ToChar(item), color);
                }
            }

            var agentCount = colors.Keys.Count(char.IsDigit);

            var initialRangeCount = (goalIndex - initialIndex) - 1;
            var initialLines = LinesToCharMatrix(serverLines.GetRange(initialIndex + 1, initialRangeCount));

            var goalRangeCount = (endIndex - goalIndex) - 1;
            var goalLines = LinesToCharMatrix(serverLines.GetRange(goalIndex + 1, goalRangeCount));

            return new Level
            {
                Colors = colors,
                AgentCount = agentCount,
                InitialMatrix = initialLines,
                GoalMatrix = goalLines
            };
        }

        private List<string> GetServerOut()
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