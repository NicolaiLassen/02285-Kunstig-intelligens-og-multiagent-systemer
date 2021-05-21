import copy
from random import randint
from typing import List

import torch
from gym.spaces import Discrete, Box, Tuple
# from ray.rllib import MultiAgentEnv

from environment.action import Action, ActionType, idxs_to_actions
from environment.level_loader import load_level
from utils.preprocess import LevelState


def debug_print(s):
    # print(s, file=sys.stderr, flush=True)
    return


class EnvWrapper():
    initial_state = None
    goal_state = None
    t0_state = None
    visited_state = None
    agents_n = 0
    rows_n = 0
    cols_n = 0
    goal_state_positions = {}
    observation_space = None
    file_name = None
    valid_agent_actions = None

    def __init__(
            self,
            env_config
    ) -> None:
        self.free_value = 32  # ord(" ")
        self.agent_0_value = 48  # ord("0")
        self.agent_9_value = 57  # ord("9")
        self.box_a_value = 65  # ord("A")
        self.box_z_value = 90  # ord("Z")

        self.num_agents = 9  # max agents
        self.action_space = Discrete(29)  # Hard code 29 actions
        self.observation_space = Tuple([
            Box(0, 100, shape=(50, 50)),  # map
            Box(0, 100, shape=(50, 50)),  # color map
            Box(0, 100, shape=(50, 50)),  # goal map
            Box(0, 100, shape=(2,))
        ])

        self.random_from_files: bool = False if 'random' not in env_config else env_config['random']

        if self.random_from_files and 'level_names' not in env_config:
            raise Exception("If random is selected you must give paths to levels")

        if self.random_from_files:
            self.level_names: List[str] = env_config['level_names']
            index = randint(0, len(self.level_names))
            self.load(index=index)
        else:
            if 'level_lines' not in env_config:
                raise Exception("If not in train add a map")
            self.load(file_lines=env_config['level_lines'])

    def __repr__(self):
        return self.t0_state.__repr__()

    def step(self, actions: dict):
        print(actions)
        actions = idxs_to_actions(list(actions.values()))
        valid_actions = []
        rewards = {}
        dones = {"__all__": False}
        for i, action in enumerate(actions):
            dones[i] = False
            rewards[i] = 0
            if not self.__is_applicable(i, action):
                rewards[i] = -0.1  # constant penalty for wrong move
                valid_actions.append(Action.NoOp)
                continue
            valid_actions.append(action)

        if self.__is_conflict(valid_actions):
            return self.__duplicate_obs(self.t0_state), rewards, dones, {}

        t1_state = self.__act(valid_actions)
        goal_count = self.__count_goals(t1_state)
        done = goal_count == len(self.goal_state_positions)
        # TODO make each reward applie to that agent
        goal_count_disc = goal_count / len(self.goal_state_positions)
        for key in rewards.keys():
            rewards[key] += goal_count_disc  # joined score for reward
            dones[key] = done

        dones["__all__"] = done
        if done:
            print(done)

        self.t0_state = t1_state
        return self.__duplicate_obs(self.t0_state), rewards, dones, {}

    def reset(self):
        if self.random_from_files:
            index = randint(0, len(self.level_names))
            self.load(index=index)
        else:
            self.t0_state = copy.deepcopy(self.initial_state)
        return self.__duplicate_obs(self.t0_state)

    def __duplicate_obs(self, state):
        obs = {}
        for i in range(self.agents_n):
            obs[i] = [state.level.numpy(),
                      state.colors.numpy(),
                      self.goal_state.level.numpy(),
                      state.agents[i].numpy()]
        return obs

    def __count_goals(self, state):
        goal_count = 0
        for row in range(len(state.level)):
            for col in range(len(state.level[row])):
                key = str([row, col])
                if key not in self.goal_state_positions:
                    continue
                val = state.level[row][col].item()
                goal_count += 1 if self.goal_state_positions[key] == val else 0
        return goal_count

    def __is_applicable(self, index: int, action: Action):
        agent_row, agent_col = self.t0_state.agent_row_col(index)

        if action.type is ActionType.NoOp:
            return True
        elif action.type is ActionType.Move:
            # check that next position is free
            next_agent_row = agent_row + action.agent_row_delta
            next_agent_col = agent_col + action.agent_col_delta
            return self.__is_free(next_agent_row, next_agent_col)
        elif action.type is ActionType.Push:
            # check that next agent position is box
            next_agent_row = agent_row + action.agent_row_delta
            next_agent_col = agent_col + action.agent_col_delta
            if not self.__is_box(next_agent_row, next_agent_col):
                debug_print('# next agent position is NOT box')
                return False
            # check that agent and box is same color
            if not self.__is_same_color(agent_row, agent_col, next_agent_row, next_agent_col):
                debug_print('# box and agent is NOT same color')
                return False
            # check that next box position is free
            next_box_row = next_agent_row + action.box_row_delta
            next_box_col = next_agent_col + action.box_col_delta
            return self.__is_free(next_box_row, next_box_col)
        elif action.type is ActionType.Pull:
            # check that next agent position is free
            next_agent_row = agent_row + action.agent_row_delta
            next_agent_col = agent_col + action.agent_col_delta
            if not self.__is_free(next_agent_row, next_agent_col):
                debug_print('# next agent position is NOT free')
                return False
            # check that box position is box
            box_row = agent_row + (action.box_row_delta * -1)
            box_col = agent_col + (action.box_col_delta * -1)
            if not self.__is_box(box_row, box_col):
                debug_print('# box position is NOT box')
                return False
            # check that agent and box is same color
            return self.__is_same_color(agent_row, agent_col, box_row, box_col)
        else:
            return False

    def __is_conflict(self, actions: List[Action]):
        num_agents = self.agents_n

        next_agent_rows = [-1 for _ in range(num_agents)]
        next_agent_cols = [-1 for _ in range(num_agents)]
        box_rows = [-1 for _ in range(num_agents)]
        box_cols = [-1 for _ in range(num_agents)]

        for i in range(num_agents):
            agent_row, agent_col = self.t0_state.agent_row_col(i)
            action = actions[i]

            if action.type is ActionType.NoOp:
                continue
            elif action.type is ActionType.Move:
                next_agent_rows[i] = agent_row + action.agent_row_delta
                next_agent_cols[i] = agent_col + action.agent_col_delta
            elif action.type is ActionType.Push:
                next_agent_rows[i] = agent_row + action.agent_row_delta
                next_agent_cols[i] = agent_col + action.agent_col_delta
                box_rows[i] = next_agent_rows[i]
                box_cols[i] = next_agent_cols[i]
            elif action.type is ActionType.Pull:
                next_agent_rows[i] = agent_row + action.agent_row_delta
                next_agent_cols[i] = agent_col + action.agent_col_delta
                box_rows[i] = agent_row + (action.box_row_delta * -1)
                box_cols[i] = agent_col + (action.box_col_delta * -1)

        for a1 in range(num_agents):
            if actions[a1].type is ActionType.NoOp:
                continue

            for a2 in range(num_agents):
                if a1 == a2:
                    continue
                if actions[a2].type is ActionType.NoOp:
                    continue

                # is moving same box
                if box_rows[a1] == box_rows[a2] and box_cols[a1] == box_cols[a2]:
                    debug_print('# actions.conflict\nis moving same box')
                    return True

                # is moving into same position
                if next_agent_rows[a1] == next_agent_rows[a2] and next_agent_cols[a1] == next_agent_cols[a2]:
                    debug_print('# actions.conflict\nis moving into same position')
                    return True

        return False

    def __is_same_color(self, a_row, a_col, b_row, b_col):
        return self.t0_state.colors[a_row][a_col] == self.t0_state.colors[b_row][b_col]

    def __is_box(self, row, col):
        return self.box_a_value <= self.t0_state.level[row][col] <= self.box_z_value

    def __is_free(self, row, col):
        return self.t0_state.level[row][col].item() == self.free_value

    def __act(self, actions: List[Action]) -> LevelState:

        next_state = self.t0_state

        for index, action in enumerate(actions):
            # Update agent location
            prev_agent_row, prev_agent_col = self.t0_state.agent_row_col(index)
            next_agent_row = prev_agent_row + action.agent_row_delta
            next_agent_col = prev_agent_col + action.agent_col_delta
            agent_value = self.t0_state.level[prev_agent_row][prev_agent_col]
            agent_color = self.t0_state.colors[prev_agent_row][prev_agent_col]
            next_state.agents[index] = torch.tensor([next_agent_row, next_agent_col])

            # Update level matrix
            if action.type is ActionType.NoOp:
                continue
            elif action.type is ActionType.Move:
                next_state.level[next_agent_row][next_agent_col] = agent_value
                next_state.level[prev_agent_row][prev_agent_col] = self.free_value

                next_state.colors[next_agent_row][next_agent_col] = agent_color
                next_state.colors[prev_agent_row][prev_agent_col] = 0

            elif action.type is ActionType.Push:
                box_value = self.t0_state.level[next_agent_row][next_agent_col]
                box_color = self.t0_state.colors[next_agent_row][next_agent_col]
                next_box_row = next_agent_row + action.box_row_delta
                next_box_col = next_agent_col + action.box_col_delta

                next_state.level[next_box_row][next_box_col] = box_value
                next_state.level[next_agent_row][next_agent_col] = agent_value
                next_state.level[prev_agent_row][prev_agent_col] = self.free_value

                next_state.colors[next_box_row][next_box_col] = box_color
                next_state.colors[next_agent_row][next_agent_col] = agent_color
                next_state.colors[prev_agent_row][prev_agent_col] = 0

            elif action.type is ActionType.Pull:
                prev_box_row = prev_agent_row + (action.box_row_delta * -1)
                prev_box_col = prev_agent_col + (action.box_col_delta * -1)
                box_value = self.t0_state.level[prev_box_row][prev_box_col]
                box_color = self.t0_state.colors[prev_box_row][prev_box_col]

                next_state.level[next_agent_row][next_agent_col] = agent_value
                next_state.level[prev_agent_row][prev_agent_col] = box_value
                next_state.level[prev_box_row][prev_box_col] = self.free_value

                next_state.colors[next_agent_row][next_agent_col] = agent_color
                next_state.colors[prev_agent_row][prev_agent_col] = box_color
                next_state.colors[prev_box_row][prev_box_col] = 0

        return next_state

    # UTILS

    def load(self, index: int = None, file_lines: List[str] = None, file_name: str = None):

        if self.random_from_files:
            file_lines, file_name = self.read_level_file(index)

        initial_state, goal_state = load_level(file_lines)
        self.file_name = file_name
        self.initial_state = initial_state
        self.goal_state = goal_state

        self.agents_n = initial_state.agents_n
        self.rows_n = initial_state.rows_n
        self.cols_n = initial_state.cols_n

        self.goal_state_positions = {}
        for row in range(len(self.goal_state.level)):
            for col in range(len(self.goal_state.level[row])):
                val = goal_state.level[row][col]
                if self.box_a_value <= val <= self.box_z_value:
                    self.goal_state_positions[str([row, col])] = val.item()
                if self.agent_0_value <= val <= self.agent_9_value:
                    self.goal_state_positions[str([row, col])] = val.item()

        self.t0_state = copy.deepcopy(initial_state)

    def read_level_file(self, index: int):
        file_name = self.level_names[index % len(self.level_names)]
        level_file = open(file_name, 'r')

        level_file_lines = [line.strip().replace("\n", "") if line.startswith("#") else line.replace("\n", "")
                            for line in level_file.readlines()]
        level_file.close()
        return level_file_lines, file_name
