using System.Collections.Generic;
using MaMapF.Models;

namespace MaMapF
{
    class Program
    {
        static void Main(string[] args)
        {
            SingleAgent();
            // var level = ServerHandler.GetServerLevel();
            // var cbs = new CBSHandler(level);
            // var plan = cbs.Search();
            // // ServerHandler.SendServerPlan(plan);
        }

        static void MultiAgent()
        {
            var level = ServerHandler.GetServerLevel();
            var cbs = new CBSHandler(level);
            var plan = cbs.Search();
            ServerHandler.SendServerPlan(plan);
        }

        static void SingleAgent()
        {
            var level = ServerHandler.GetServerLevel();
            var lowLevel = new LowLevelSearch(level);
            var plan = lowLevel.GetSingleAgentPlan('0', new List<Constraint>());
            ServerHandler.SendServerPlan(new Dictionary<char, List<SingleAgentState>>
            {
                {'0', plan}
            });
        }
    }
}