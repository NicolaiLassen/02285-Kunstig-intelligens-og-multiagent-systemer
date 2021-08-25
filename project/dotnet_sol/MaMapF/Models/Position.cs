namespace MaMapF.Models
{
    public class Position
    {
        public int Row { get; set; }
        public int Column { get; set; }

        public Position(int row, int column)
        {
            this.Row = row;
            this.Column = column;
        }

        public Position Next(int rowDelta, int colDelta)
        {
            return new Position(
                this.Row + rowDelta,
                this.Column + colDelta
            );
        }
    }
}