import os

import numpy as np
import ray
from ray.rllib.agents.ppo import ppo
from ray.tune import register_env

from environment.env_wrapper import EnvWrapper


def absolute_file_paths(directory):
    path = os.path.abspath(directory)
    return [entry.path for entry in os.scandir(path) if entry.is_file()]


if __name__ == '__main__':
    # BEFORE SERVER
    ray.init(include_dashboard=False)

    level_file = open('./levels_manual/N1.lvl', 'r')
    level_file_lines = [line.strip().replace("\n", "") if line.startswith("#") else line.replace("\n", "")
                        for line in level_file.readlines()]
    level_file.close()

    # server here

    env = EnvWrapper({'level_lines': level_file_lines})
    env_name = "multi_agent_env"


    def env_creator(_):
        return env


    register_env(env_name, env_creator)

    agent = ppo.PPOTrainer(env='multi_agent_env',
                           config={
                               "in_evaluation": True,
                               "num_workers": 0,
                               "model": {
                                   "use_lstm": True,
                                   "max_seq_len": 100,
                                   "lstm_cell_size": 256,
                                   "conv_filters": None,
                                   "conv_activation": "relu",
                                   "num_framestacks": 0
                               },
                               "log_level": "ERROR",
                               "framework": "torch"
                           })

    agent.restore('./ckpt/checkpoint_000501/checkpoint-501')
    final_actions = []
    s1 = env.reset()
    r1 = None
    a1 = None

    state = {}
    for i in range(env.num_agents):
        state[i] = [np.zeros(256, np.float32), np.zeros(256, np.float32)]

    for i in range(100):

        s = s1
        actions, state, _ = agent.compute_actions(s, state=state, prev_action=a1)

        s1, r1, d, _ = env.step(actions)
        a1 = actions

        if (s[0][0] == s1[0][0]).all():
            continue

        final_actions.append(actions)

        if d['__all__']:
            break

    print(final_actions)
