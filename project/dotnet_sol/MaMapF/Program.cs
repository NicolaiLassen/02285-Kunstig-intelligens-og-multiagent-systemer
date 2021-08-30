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
                throw new NullReferenceException("Main.solution == null");
            }

            ServerHandler.SendServerPlan(solution);
        }
    }
}