namespace MaMapF.Models
{
    public class MapItem
    {
        public char Value { get; set; }
        public Position Position { get; set; }

        public MapItem(char value, Position position)
        {
            this.Value = value;
            this.Position = position;
        }

        public override string ToString()
        {
            return $"MAPITEM\n Value: {Value}, Position: {Position}";
        }

        public bool Equals(MapItem other)
        {
            return Value == other.Value && Position.Equals(other.Position);
        }
    }
}