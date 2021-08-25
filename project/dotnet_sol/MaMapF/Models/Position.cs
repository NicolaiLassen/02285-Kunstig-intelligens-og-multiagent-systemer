namespace MaMapF.Models
{

    public interface IPosition
    {
        int Row { get; set; }
        int Column { get; set; }
    }
    
    public class Position: IPosition
    {
        public int Row { get; set; }
        public int Column { get; set; }

        public Position(int row, int column)
        {
            Row = row;
            Column = column;
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