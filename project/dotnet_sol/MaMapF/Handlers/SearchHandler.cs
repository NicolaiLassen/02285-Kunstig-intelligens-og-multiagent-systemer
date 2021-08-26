using System.Collections.Generic;
using MaMapF.Models;

namespace MaMapF.Handlers
{
    public class SearchHandler
    {

        private Level Level;
        
        public SearchHandler(Level level)
        {
            Level = level;
        }
        
        public Dictionary<char, List<SingleAgentState>> Search()
        {
            
            
            
            return CBSHandler.Search(Level.Agents, Level.AgentInitialStates, Level.Goals);
        } 
    }
}