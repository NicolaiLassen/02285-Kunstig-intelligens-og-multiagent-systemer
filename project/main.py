import os

import ray
from ray.tune import register_env, tune, grid_search

from environment.env_wrapper import EnvWrapper


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def absolute_file_paths(directory):
    path = os.path.abspath(directory)
    return [entry.path for entry in os.scandir(path) if entry.is_file()]


if __name__ == '__main__':
    ray.init()
    create_dir('./ckpt')


    def env_creator(_):
        level_file_paths = absolute_file_paths('./levels_manual')
        return EnvWrapper({'random': True, 'level_names': level_file_paths})


    env_name = "multi_agent_env"
    register_env(env_name, env_creator)

    analysis = tune.run("PPO",
                        local_dir='./ckpt',
                        checkpoint_at_end=True,
                        stop={"training_iteration": 1000},
                        config={
                            "env": "multi_agent_env",
                            "horizon": 1000,
                            "num_gpus": 1,
                            "num_workers": 6,
                            "framework": "torch",
                            "lr": grid_search([0.0001]),
                            "model": {
                                "conv_filters": None,
                                "conv_activation": "relu",
                                "num_framestacks": 0
                            }
                        })
