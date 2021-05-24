import argparse
import os

from agents.mappo_trainer import MAPPOTrainer
from environment.env_wrapper import MultiAgentEnvWrapper
from models.policy_models import ActorPolicyModel, CriticPolicyModel


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
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--ckpt", default="ckpt", help="ckpt name")
    args = parser.parse_args()

    create_dir('./{}'.format(args.ckpt))

    level_file_paths_man = absolute_file_paths('./levels_comp')

    width = 50
    height = 50

    env_wrapper = MultiAgentEnvWrapper({'random': True, 'level_names': level_file_paths_man})
    actor = ActorPolicyModel(width, height, env_wrapper.action_space_n).cuda()
    critic = CriticPolicyModel(width, height).cuda()

    agent_trainer = MAPPOTrainer(
        actor,
        critic,
        env_wrapper
    )

    agent_trainer.train(int(200e6))
