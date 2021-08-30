using System;

namespace MaMapF.Models
{
    public class Constraint
    {
        public char Agent { get; set; }
        public Position Position { get; set; }

        public int Step { get; set; }

        public override string ToString()
        {
            return $"CONSTRAINT\nAgent: {Agent}, Position: {Position}, Step: {Step}";
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

        public override int GetHashCode()
        {
            return HashCode.Combine(Agent, Position, Step);
        }

        public Constraint Copy()
        {
            return new Constraint
            {
                Agent = Agent,
                Position = Position.Next(0, 0),
                Step = Step + 0,
            };
        }
    }
}