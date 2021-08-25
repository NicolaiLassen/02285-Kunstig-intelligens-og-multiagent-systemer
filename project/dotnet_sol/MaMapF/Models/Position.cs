namespace MaMapF.Models
{
    public class Position
    {
        public int Row { get; set; }
        public int Column { get; set; }

        public Position()
        {
        }

        public Position(int row, int column)
        {
            Row = row;
            Column = column;
        }

        public Position Next(int rowDelta, int colDelta)
        {
            return new Position(
                Row + rowDelta,
                Column + colDelta
            );
        }


        public override string ToString()
        {
            return $"{Row},{Column}";
        }

        // TODO?
        // public override int GetHashCode() => HashCode.Combine(Row, Column);
        //
        // public override bool Equals(object obj)
        // {
        //     if (!(obj is Position other)) return false;
        //     return Row == other.Row && Column == other.Column;
        // }
    }
}