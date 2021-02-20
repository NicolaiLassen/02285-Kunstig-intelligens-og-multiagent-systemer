package searchclient;

public class Memory
{
    private static final Runtime RUNTIME = Runtime.getRuntime();
    private static final double MB = 1024 * 1024;

    public static double used()
    {
        return (RUNTIME.totalMemory() - RUNTIME.freeMemory()) / MB;
    }

    public static double free()
    {
        return RUNTIME.freeMemory() / MB;
    }

    public static double total()
    {
        return RUNTIME.totalMemory() / MB;
    }

    public static double max()
    {
        return RUNTIME.maxMemory() / MB;
    }

    public static String stringRep()
    {
        return String.format("[Used: %4.2f MB, Free: %4.2f MB, Alloc: %4.2f MB, MaxAlloc: %4.2f MB]",
                             used(),
                             free(),
                             total(),
                             max());
    }
}
