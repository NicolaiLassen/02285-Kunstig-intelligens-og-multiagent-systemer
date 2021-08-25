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


        public override string ToString()
        {
            return $"{Row},{Column}";
        }


        public bool Equals(Position other)
        {
            return this.Row == other.Row && this.Column == other.Column;
        }

        // public override bool Equals(object? obj)
        // {
        //     if (obj == null) return false;
        //     if (!(obj is Position other)) return false;
        //     return this.Equals(other);
        // }

        // public override int GetHashCode()
        // {
        //     return HashCode.Combine(Row, Column);
        // }
        //
        // public static bool operator ==(Position left, Position right)
        // {
        //     if (left == null || right == null) return false;
        //     return left.Equals(right);
        // }
        //
        // public static bool operator !=(Position left, Position right)
        // {
        //     return !(left == right);
        // }


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