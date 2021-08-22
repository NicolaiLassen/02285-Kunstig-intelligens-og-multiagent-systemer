from copy import deepcopy

from environment.action import Action, ActionType
from michael.a_state import AState


def is_box(state, row, col) -> bool:
    return "A" <= state.map[row][col] <= "Z"


def is_free(state, row, col) -> bool:
    return state.map[row][col] == " "


def is_applicable(state: AState, action: Action) -> bool:
    agent_row = state.agent_row
    agent_col = state.agent_col

    if action.type is ActionType.NoOp:
        return True
    elif action.type is ActionType.Move:
        # check that next position is free
        next_agent_row = agent_row + action.agent_row_delta
        next_agent_col = agent_col + action.agent_col_delta
        return is_free(state, next_agent_row, next_agent_col)
    elif action.type is ActionType.Push:
        # check that next agent position is box
        next_agent_row = agent_row + action.agent_row_delta
        next_agent_col = agent_col + action.agent_col_delta
        if not is_box(state, next_agent_row, next_agent_col):
            return False
        # check that next box position is free
        next_box_row = next_agent_row + action.box_row_delta
        next_box_col = next_agent_col + action.box_col_delta
        return is_free(state, next_box_row, next_box_col)
    elif action.type is ActionType.Pull:
        # check that next agent position is free
        next_agent_row = agent_row + action.agent_row_delta
        next_agent_col = agent_col + action.agent_col_delta
        if not is_free(state, next_agent_row, next_agent_col):
            return False
        # check that box position is box
        box_row = agent_row + (action.box_row_delta * -1)
        box_col = agent_col + (action.box_col_delta * -1)
        if not is_box(state, box_row, box_col):
            return False

        return True
    else:
        return False


def act(state, action: Action) -> AState:
    next_state = deepcopy(state)

    # Update agent location
    prev_agent_row, prev_agent_col = state.agent_row
    next_agent_row = prev_agent_row + action.agent_row_delta
    next_agent_col = prev_agent_col + action.agent_col_delta
    agent_value = state.map[prev_agent_row][prev_agent_col]

    next_state.agent_row = next_agent_row
    next_state.agent_col = next_agent_col

    # Update level matrices and agent pos
    if action.type is ActionType.NoOp:
        return next_state
    elif action.type is ActionType.Move:
        next_state.map[next_agent_row][next_agent_col] = agent_value
        next_state.map[prev_agent_row][prev_agent_col] = " "

    elif action.type is ActionType.Push:
        box_value = state.map[next_agent_row][next_agent_col]
        next_box_row = next_agent_row + action.box_row_delta
        next_box_col = next_agent_col + action.box_col_delta

        next_state.map[next_box_row][next_box_col] = box_value
        next_state.map[next_agent_row][next_agent_col] = agent_value
        next_state.map[prev_agent_row][prev_agent_col] = " "


    elif action.type is ActionType.Pull:
        prev_box_row = prev_agent_row + (action.box_row_delta * -1)
        prev_box_col = prev_agent_col + (action.box_col_delta * -1)
        box_value = state.map[prev_box_row][prev_box_col]

        next_state.map[next_agent_row][next_agent_col] = agent_value
        next_state.map[prev_agent_row][prev_agent_col] = box_value
        next_state.map[prev_box_row][prev_box_col] = " "

    # next_state.parent = self
    next_state.action = action
    next_state.g = state.g + 1

    return next_state


def expand_state(state: AState):
    applicable_actions = [action for action in Action if is_applicable(state, action)]
    return [act(state, action) for action in applicable_actions]
