using System;

namespace MaMapF.Models
{
    public interface IPosition
    {
        int Row { get; set; }
        int Column { get; set; }
    }

    public class Position : IPosition
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

        public override int GetHashCode() => HashCode.Combine(Row, Column);

        public override bool Equals(object obj)
        {
            if (obj == null) return false;
            if (!(obj is Position other)) return false;
            return Row == other.Row && Column == other.Column;
        }
    }
}