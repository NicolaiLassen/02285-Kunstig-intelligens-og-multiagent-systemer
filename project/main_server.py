import os

import ray
from ray.rllib.agents.ppo import ppo
from ray.tune import register_env

from environment.env_wrapper import EnvWrapper


def absolute_file_paths(directory):
    path = os.path.abspath(directory)
    return [entry.path for entry in os.scandir(path) if entry.is_file()]


if __name__ == '__main__':
    # BEFORE SERVER
    ray.init()
    level_file = open('./levels_manual/N0.lvl', 'r')
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
                               "log_level": "ERROR",
                               "framework": "torch"
                           })

    # agent.restore('./ckpt/checkpoint_000401/checkpoint-401')
    final_actions = []
    s1 = env.reset()
    r1 = None
    a1 = None
    for i in range(10):
        s = s1
        actions = agent.compute_actions(s, prev_reward=r1, prev_action=a1)
        s1, r, d, _ = env.step(actions)
        print(s1)
        if (s[0][0]==s1[0][0]).all():
            continue
        actions_s1 = actions
        final_actions.append(actions)
        print(r)
        if d['__all__']:
            print(d)
            break

    print(final_actions)
