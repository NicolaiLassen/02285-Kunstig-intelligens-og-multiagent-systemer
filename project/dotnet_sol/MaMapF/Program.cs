using System;
using MaMapF.Handlers;

namespace MaMapF
{
    class Program
    {
        static void Main(string[] args)
        {
            var level = ServerHandler.GetServerLevel();
            var searchHandler = new SearchHandler(level);
            var solution = searchHandler.Search();
            
            if (solution == null)
            {
                Console.Error.WriteLine("SOLUTION WAS NULL");
                Environment.Exit(0);
            }


            ServerHandler.SendServerPlan(solution);
        }
    }
}