package searchclient;

enum ActionType {
    NoOp,
    Move,
    Push,
    Pull
}

public enum Action {
    // push box no wall
    //    MoveN("Move(N)", ActionType.Move, -1, 0, 0, 0),
    PushNN("Push(N,N)", ActionType.Push, -1, 0, -1, 0),
    // push box wall
    //    MoveW("Move(W)", ActionType.Move, 0, -1, 0, 0),
    PushNW("Push(N,W)", ActionType.Push, -1, 0, 0, -1),
    //    MoveE("Move(E)", ActionType.Move, 0, 1, 0, 0),
    PushNE("Push(N,E)", ActionType.Push, -1, 0, 0, 1),

    //    MoveS("Move(S)", ActionType.Move, 1, 0, 0, 0),
    PushSS("Push(S,S)", ActionType.Push, 1, 0, 1, 0),
    //    MoveW("Move(W)", ActionType.Move, 0, -1, 0, 0),
    PushSW("Push(S,W)", ActionType.Push, 1, 0, 0, -1),
    //    MoveE("Move(E)", ActionType.Move, 0, 1, 0, 0),
    PushSE("Push(S,E)", ActionType.Push, 1, 0, 0, 1),

    //    MoveE("Move(E)", ActionType.Move, 0, 1, 0, 0),
    PushEE("Push(E,E)", ActionType.Push, 0, 1, 0, 1),
    //    MoveN("Move(N)", ActionType.Move, -1, 0, 0, 0),
    PushEN("Push(E,N)", ActionType.Push, 0, 1, -1, 0),
    //    MoveS("Move(S)", ActionType.Move, 1, 0, 0, 0),
    PushES("Push(E,S)", ActionType.Push, 0, 1, 1, 0),

    //    MoveW("Move(W)", ActionType.Move, 0, -1, 0, 0),
    PushWW("Push(W,W)", ActionType.Push, 0, -1, 0, -1),
    //    MoveS("Move(S)", ActionType.Move, 1, 0, 0, 0),
    PushWS("Push(W,S)", ActionType.Push, 0, -1, 1, 0),
    //    MoveN("Move(N)", ActionType.Move, -1, 0, 0, 0),
    PushWN("Push(W,N)", ActionType.Push, 0, -1, -1, 0),

    // pull box no wall
    //    MoveN("Move(N)", ActionType.Move, -1, 0, 0, 0),
    PullNN("Pull(N,N)", ActionType.Pull, -1, 0, -1, 0),
    // pull box wall
    //    MoveW("Move(W)", ActionType.Move, 0, -1, 0, 0),
    PullNW("Pull(N,W)", ActionType.Pull, -1, 0, 0, -1),
    //    MoveE("Move(E)", ActionType.Move, 0, 1, 0, 0),
    PullNE("Pull(N,E)", ActionType.Pull, -1, 0, 0, 1),

    //    MoveS("Move(S)", ActionType.Move, 1, 0, 0, 0),
    PullSS("Pull(S,S)", ActionType.Pull, 1, 0, 1, 0),
    //    MoveW("Move(W)", ActionType.Move, 0, -1, 0, 0),
    PullSW("Pull(S,W)", ActionType.Pull, 1, 0, 0, -1),
    //    MoveE("Move(E)", ActionType.Move, 0, 1, 0, 0),
    PullSE("Pull(S,E)", ActionType.Pull, 1, 0, 0, 1),

    //    MoveE("Move(E)", ActionType.Move, 0, 1, 0, 0),
    PullEE("Pull(E,E)", ActionType.Pull, 0, 1, 0, 1),
    //    MoveN("Move(N)", ActionType.Move, -1, 0, 0, 0),
    PullEN("Pull(E,N)", ActionType.Pull, 0, 1, -1, 0),
    //    MoveS("Move(S)", ActionType.Move, 1, 0, 0, 0),
    PullES("Pull(E,S)", ActionType.Pull, 0, 1, 1, 0),

    //    MoveW("Move(W)", ActionType.Move, 0, -1, 0, 0),
    PullWW("Pull(W,W)", ActionType.Pull, 0, -1, 0, -1),
    //    MoveS("Move(S)", ActionType.Move, 1, 0, 0, 0),
    PullWS("Pull(W,S)", ActionType.Pull, 0, -1, 1, 0),
    //    MoveN("Move(N)", ActionType.Move, -1, 0, 0, 0),
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
