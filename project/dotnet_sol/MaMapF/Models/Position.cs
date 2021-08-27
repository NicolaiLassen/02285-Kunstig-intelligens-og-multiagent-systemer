using System;

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

        public static int Distance(MapItem a, MapItem b)
        {
            return Distance(a.Position, b.Position);
        }

        public static int Distance(Position a, Position b)
        {
            return Math.Abs(a.Row - b.Row) + Math.Abs(a.Column - b.Column);
        }

        public override string ToString()
        {
            return $"{Row},{Column}";
        }

        public bool Equals(Position other)
        {
            return Row == other.Row && Column == other.Column;
        }
    }
}