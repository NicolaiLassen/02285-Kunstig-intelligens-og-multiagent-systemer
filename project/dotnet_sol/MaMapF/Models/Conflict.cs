namespace MaMapF.Models
{
    public class Conflict
    {
        public string Type { get; set; }
        public char AgentA { get; set; }
        public char AgentB { get; set; }
        public Position Position { get; set; }
        public int Step { get; set; }

        public override string ToString()
        {
            return $"CONFLICT: Agent: {AgentA} v {AgentB}, position: {Position}, step: {Step}\n";
        }
    }
}