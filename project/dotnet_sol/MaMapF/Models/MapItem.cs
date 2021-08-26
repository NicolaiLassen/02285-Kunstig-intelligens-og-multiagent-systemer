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
    }
}