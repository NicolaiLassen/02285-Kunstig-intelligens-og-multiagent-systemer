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

        static void SingleAgent()
        {
            var level = ServerHandler.GetServerLevel();
            var lowLevel = new LowLevelSearch {Level = level};
            var initialState = level.GetInitialState('0');
            var plan = lowLevel.GetSingleAgentPlan(initialState, new List<Constraint>());
            ServerHandler.SendServerPlan(plan);
        }
    }
}