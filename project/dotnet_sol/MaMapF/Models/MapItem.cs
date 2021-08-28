using System;

namespace MaMapF.Models
{
    public class MapItem
    {
        public string UID { get; }
        public char Value { get; }
        public Position Position { get; set; }

        public MapItem(string uid, char value, Position position)
        {
            UID = uid;
            Value = value;
            Position = position;
        }

        public MapItem(char value, Position position)
        {
            Value = value;
            Position = position;
            UID = $"{value}.{position.Row}.{position.Column}";
        }


        public override bool Equals(object obj)
        {
            if (obj == null) return false;
            if (!(obj is MapItem other)) return false;
            return Position.Equals(other.Position) && Value == other.Value;
        }

        public override int GetHashCode()
        {
            return HashCode.Combine(UID, Value, Position.GetHashCode());
        }


        public override string ToString()
        {
            return $"MAPITEM {Value} at {Position} | {UID}";
        }

        public bool Equals(MapItem other)
        {
            return Value == other.Value && Position.Equals(other.Position);
        }
    }
}