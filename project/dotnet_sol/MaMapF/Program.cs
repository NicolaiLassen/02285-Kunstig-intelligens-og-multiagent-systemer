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
            var lowLevelSearch = new LowLevelSearch
            {
                Level = level
            };

            var plan = lowLevelSearch.GetSingleAgentPlan(initialState, new List<Constraint>());
            
            ParserHandler.SendServerPlan(plan);
        }
    }
}