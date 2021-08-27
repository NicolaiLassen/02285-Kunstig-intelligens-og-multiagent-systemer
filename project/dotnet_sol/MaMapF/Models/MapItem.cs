namespace MaMapF.Models
{
    public class MapItem
    {
        public char Value { get; set; }
        public Position Position { get; set; }

        public MapItem(char value, Position position)
        {
            Value = value;
            Position = position;
        }


        public override bool Equals(object obj)
        {
            if (obj == null) return false;
            if (!(obj is MapItem other)) return false;
            return Position.Equals(other.Position) && Value == other.Value;
        }


        public override string ToString()
        {
            return $"MAPITEM\n Value: {Value}, Position: {Position}";
        }
    }
}