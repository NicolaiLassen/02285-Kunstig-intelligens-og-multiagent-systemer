namespace MaMapF.Models
{
    public class Constraint
    {
        public char Agent { get; set; }
        public Position Position { get; set; }

        public int Step { get; set; }
        // public Conflict Conflict { get; set; }

        public override string ToString()
        {
            return $"CONSTRAINT\nAgent: {Agent}, Position: {Position}, Step: {Step}\n";
        }
    }
}