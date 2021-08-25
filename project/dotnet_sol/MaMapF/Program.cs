namespace MaMapF
{
    class Program
    {
        static void Main(string[] args)
        {
            var level = ServerHandler.GetServerLevel();
            var cbs = new CBSHandler(level);
            var plan = cbs.Search();
            ServerHandler.SendServerPlan(plan);
        }
    }
}