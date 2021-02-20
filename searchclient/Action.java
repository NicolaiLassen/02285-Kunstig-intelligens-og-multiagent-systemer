package searchclient;

enum ActionType {
    NoOp,
    Move,
    Push,
    Pull
}

public enum Action {
    // push box no wall
    PushNN("Push(N,N)", ActionType.Push, -1, 0, -1, 0),
    // push box wall
    PushNW("Push(N,W)", ActionType.Push, -1, 0, 0, -1),
    PushNE("Push(N,E)", ActionType.Push, -1, 0, 0, 1),

    PushSS("Push(S,S)", ActionType.Push, 1, 0, 1, 0),
    PushSW("Push(S,W)", ActionType.Push, 1, 0, 0, -1),
    PushSE("Push(S,E)", ActionType.Push, 1, 0, 0, 1),

    PushEE("Push(E,E)", ActionType.Push, 0, 1, 0, 1),
    PushEN("Push(E,N)", ActionType.Push, 0, 1, -1, 0),
    PushES("Push(E,S)", ActionType.Push, 0, 1, 1, 0),

    PushWW("Push(W,W)", ActionType.Push, 0, -1, 0, -1),
    PushWS("Push(W,S)", ActionType.Push, 0, -1, 1, 0),
    PushWN("Push(W,N)", ActionType.Push, 0, -1, -1, 0),

    // pull box no wall
    PullNN("Pull(N,N)", ActionType.Pull, -1, 0, -1, 0),
    // pull box wall
    PullNW("Pull(N,W)", ActionType.Pull, -1, 0, 0, -1),
    PullNE("Pull(N,E)", ActionType.Pull, -1, 0, 0, 1),

    PullSS("Pull(N,N)", ActionType.Pull, 1, 0, 1, 0),
    PullSW("Pull(S,W)", ActionType.Pull, 1, 0, 0, -1),
    PullSE("Pull(S,E)", ActionType.Pull, 1, 0, 0, 1),

    PullEE("Pull(E,E)", ActionType.Pull, 0, 1, 1, 0),
    PullEN("Pull(E,N)", ActionType.Pull, 0, 1, -1, 0),
    PullES("Pull(E,S)", ActionType.Pull, 0, 1, 1, 0),

    PullWW("Pull(W,W)", ActionType.Pull, 0, -1, 0, -1),
    PullWS("Pull(W,S)", ActionType.Pull, 0, -1, 1, 0),
    PullWN("Pull(W,N)", ActionType.Pull, 0, -1, -1, 0),

    NoOp("NoOp", ActionType.NoOp, 0, 0, 0, 0),

    MoveN("Move(N)", ActionType.Move, -1, 0, 0, 0),
    MoveS("Move(S)", ActionType.Move, 1, 0, 0, 0),
    MoveE("Move(E)", ActionType.Move, 0, 1, 0, 0),
    MoveW("Move(W)", ActionType.Move, 0, -1, 0, 0);

    public final String name;
    public final ActionType type;
    public final int agentRowDelta; // vertical displacement of agent (-1,0,+1)
    public final int agentColDelta; // horisontal displacement of agent (-1,0,+1)
    public final int boxRowDelta; // vertical diplacement of box (-1,0,+1)
    public final int boxColDelta; // horisontal displacement of box (-1,0,+1)

    Action(String name, ActionType type, int ard, int acd, int brd, int bcd) {
        this.name = name;
        this.type = type;
        this.agentRowDelta = ard;
        this.agentColDelta = acd;
        this.boxRowDelta = brd;
        this.boxColDelta = bcd;
    }
}
