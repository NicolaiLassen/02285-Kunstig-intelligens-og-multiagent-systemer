﻿using System.Collections.Generic;

namespace MaMapF.Models
{
    public class Conflict
    {
        public string Type { get; set; }
        public char AgentA { get; set; }
        public char AgentB { get; set; }
        public List<Position> Position { get; set; }
        public int Step { get; set; }
    }
}