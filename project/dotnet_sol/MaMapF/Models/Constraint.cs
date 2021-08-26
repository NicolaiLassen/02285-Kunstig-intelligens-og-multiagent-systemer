using System;

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

        public bool Equals(Constraint other)
        {
            return Agent == other.Agent && Position.Equals(other.Position) && Step == other.Step;
        }

        public override bool Equals(object? obj)
        {
            if (obj == null) return false;
            if (!(obj is Constraint other)) return false;
            return Equals(other);
        }

        // public override int GetHashCode()
        // {
        //     return 31 * Agent + Position.GetHashCode() + 23 * Step;
        //     return HashCode.Combine(Agent, Position, Step);
        // }
    }
}