namespace MaMapF.Models
{
    public class Goal : IPosition
    {
        public char Item { get; set; }
        public int Row { get; set; }
        public int Column { get; set; }
    }
}