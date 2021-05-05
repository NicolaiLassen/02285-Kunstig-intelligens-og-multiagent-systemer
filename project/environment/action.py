from enum import Enum, unique
from typing import List

from torch import Tensor


@unique
class ActionType(Enum):
    NoOp = 0
    Move = 1
    Push = 2
    Pull = 3


@unique
class Action(Enum):
    NoOp = ("NoOp", ActionType.NoOp, 0, 0, 0, 0)

    MoveN = ("Move(N)", ActionType.Move, -1, 0, 0, 0)
    MoveS = ("Move(S)", ActionType.Move, 1, 0, 0, 0)
    MoveE = ("Move(E)", ActionType.Move, 0, 1, 0, 0)
    MoveW = ("Move(W)", ActionType.Move, 0, -1, 0, 0)

    PushNN = ("Push(N,N)", ActionType.Push, -1, 0, -1, 0)
    PushNW = ("Push(N,W)", ActionType.Push, -1, 0, 0, -1)
    PushNE = ("Push(N,E)", ActionType.Push, -1, 0, 0, 1)

    PushSS = ("Push(S,S)", ActionType.Push, 1, 0, 1, 0)
    PushSW = ("Push(S,W)", ActionType.Push, 1, 0, 0, -1)
    PushSE = ("Push(S,E)", ActionType.Push, 1, 0, 0, 1)

    PushEE = ("Push(E,E)", ActionType.Push, 0, 1, 0, 1)
    PushEN = ("Push(E,N)", ActionType.Push, 0, 1, -1, 0)
    PushES = ("Push(E,S)", ActionType.Push, 0, 1, 1, 0)

    PushWW = ("Push(W,W)", ActionType.Push, 0, -1, 0, -1)
    PushWS = ("Push(W,S)", ActionType.Push, 0, -1, 1, 0)
    PushWN = ("Push(W,N)", ActionType.Push, 0, -1, -1, 0)

    PullNN = ("Pull(N,N)", ActionType.Pull, -1, 0, -1, 0)
    PullNW = ("Pull(N,W)", ActionType.Pull, -1, 0, 0, -1)
    PullNE = ("Pull(N,E)", ActionType.Pull, -1, 0, 0, 1)

    PullSS = ("Pull(S,S)", ActionType.Pull, 1, 0, 1, 0)
    PullSW = ("Pull(S,W)", ActionType.Pull, 1, 0, 0, -1)
    PullSE = ("Pull(S,E)", ActionType.Pull, 1, 0, 0, 1)

    PullEE = ("Pull(E,E)", ActionType.Pull, 0, 1, 0, 1)
    PullEN = ("Pull(E,N)", ActionType.Pull, 0, 1, -1, 0)
    PullES = ("Pull(E,S)", ActionType.Pull, 0, 1, 1, 0)

    PullWW = ("Pull(W,W)", ActionType.Pull, 0, -1, 0, -1)
    PullWS = ("Pull(W,S)", ActionType.Pull, 0, -1, 1, 0)
    PullWN = ("Pull(W,N)", ActionType.Pull, 0, -1, -1, 0)

    def __init__(self, name, type, ard, acd, brd, bcd):
        self.name_ = name
        self.type = type
        self.agent_row_delta = ard  # horisontal displacement agent
        self.agent_col_delta = acd  # vertical displacement agent
        self.box_row_delta = brd  # horisontal displacement box
        self.box_col_delta = bcd  # vertical displacement box


action_dict = {
    0: Action.NoOp,
    1: Action.MoveN,
    2: Action.MoveS,
    3: Action.MoveE,
    4: Action.MoveW,

    5: Action.PushNN,
    6: Action.PushNE,
    7: Action.PushNW,
    8: Action.PushSS,
    9: Action.PushSE,
    10: Action.PushSW,
    11: Action.PushEE,
    12: Action.PushEN,
    13: Action.PushES,
    14: Action.PushWW,
    15: Action.PushWN,
    16: Action.PushWS,

    17: Action.PullNN,
    18: Action.PullNE,
    19: Action.PullNW,
    20: Action.PullSS,
    21: Action.PullSE,
    22: Action.PullSW,
    23: Action.PullEE,
    24: Action.PullEN,
    25: Action.PullES,
    26: Action.PullWW,
    27: Action.PullWN,
    28: Action.PullWS,
}


def idxs_to_actions(actions: List[Tensor]) -> List[Action]:
    return [action_dict[a.item()] for a in actions]
