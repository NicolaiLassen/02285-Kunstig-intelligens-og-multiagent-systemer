using System;
using System.Collections.Generic;

namespace MaMapF.Models
{
    public class Position
    {
        public int Row { get; }
        public int Column { get; }

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

        public override bool Equals(object obj)
        {
            if (obj == null) return false;
            if (!(obj is Position other)) return false;
            return Equals(other);
        }

        public override int GetHashCode()
        {
            return HashCode.Combine(Row, Column);
        }

        public static List<Position> GetNeighbours(Position p)
        {
            return new List<Position>
            {
                new Position(p.Row - 1, p.Column),
                new Position(p.Row + 1, p.Column),
                new Position(p.Row, p.Column + 1),
                new Position(p.Row, p.Column + -1),
            };
        }
    }
}