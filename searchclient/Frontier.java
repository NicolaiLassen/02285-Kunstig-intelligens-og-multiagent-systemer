package searchclient;

import java.util.ArrayDeque;
import java.util.HashSet;

public interface Frontier
{
    void add(State state);
    State pop();
    boolean isEmpty();
    int size();
    boolean contains(State state);
    String getName();
}

class FrontierBFS
        implements Frontier
{
    private final ArrayDeque<State> queue = new ArrayDeque<>(65536);
    private final HashSet<State> set = new HashSet<>(65536);

    @Override
    public void add(State state)
    {
        this.queue.addLast(state);
        this.set.add(state);
    }

    @Override
    public State pop()
    {
        State state = this.queue.pollFirst();
        this.set.remove(state);
        return state;
    }

    @Override
    public boolean isEmpty()
    {
        return this.queue.isEmpty();
    }

    @Override
    public int size()
    {
        return this.queue.size();
    }

    @Override
    public boolean contains(State state)
    {
        return this.set.contains(state);
    }

    @Override
    public String getName()
    {
        return "breadth-first search";
    }
}

class FrontierDFS
        implements Frontier
{
    @Override
    public void add(State state)
    {
        throw new NotImplementedException();
    }

    @Override
    public State pop()
    {
        throw new NotImplementedException();
    }

    @Override
    public boolean isEmpty()
    {
        throw new NotImplementedException();
    }

    @Override
    public int size()
    {
        throw new NotImplementedException();
    }

    @Override
    public boolean contains(State state)
    {
        throw new NotImplementedException();
    }

    @Override
    public String getName()
    {
        return "depth-first search";
    }
}

class FrontierBestFirst
        implements Frontier
{
    private Heuristic heuristic;

    public FrontierBestFirst(Heuristic h)
    {
        this.heuristic = h;
    }

    @Override
    public void add(State state)
    {
        throw new NotImplementedException();
    }

    @Override
    public State pop()
    {
        throw new NotImplementedException();
    }

    @Override
    public boolean isEmpty()
    {
        throw new NotImplementedException();
    }

    @Override
    public int size()
    {
        throw new NotImplementedException();
    }

    @Override
    public boolean contains(State state)
    {
        throw new NotImplementedException();
    }

    @Override
    public String getName()
    {
        return String.format("best-first search using %s", this.heuristic.toString());
    }
}
