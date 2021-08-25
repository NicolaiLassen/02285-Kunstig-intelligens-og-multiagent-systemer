using System.Collections.Generic;
using MaMapF.Models;

namespace MaMapF
{
    class Program
    {
        static void Main(string[] args)
        {
            var level = ParserHandler.GetServerLevel();

            var initialState = level.GetAgentInitialState('0');
            var plan = LowLevelSearch.GetSingleAgentPlan(level, initialState, new List<Constraint>());

            ParserHandler.SendServerPlan(plan);
        }
    }
}