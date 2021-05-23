import argparse
import os
from copy import deepcopy

import torch
from machin.frame.algorithms import MADDPG
from torch import nn

from environment.env_wrapper import MultiAgentEnvWrapper
from models.curiosity import IntrinsicCuriosityModule
from models.policy_models import CriticPolicyModel, ActorPolicyModel


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

    level_file_paths_man = absolute_file_paths('./levels_manual')

    width = 50
    height = 50
    lr_actor = 3e-4
    lr_critic = 1e-3
    lr_icm = 1e-3

    agent_num = 10

    env_wrapper = MultiAgentEnvWrapper({'random': True, 'level_names': level_file_paths_man})
    actor = ActorPolicyModel(width, height, env_wrapper.action_space_n).cuda()
    critic = CriticPolicyModel(width, height).cuda()
    icm = IntrinsicCuriosityModule(env_wrapper.action_space_n).cuda()

    optimizer = torch.optim.Adam([
        {'params': actor.parameters(), 'lr': lr_actor},
        {'params': icm.parameters(), 'lr': lr_icm},
        {'params': critic.parameters(), 'lr': lr_critic}
    ])


    agent_trainer = MADDPG(
        [deepcopy(actor) for _ in range(agent_num)],
        [deepcopy(actor) for _ in range(agent_num)],
        [deepcopy(critic) for _ in range(agent_num)],
        [deepcopy(critic) for _ in range(agent_num)],
        critic_visible_actors=[list(range(agent_num))] * agent_num,
        optimizer=torch.optim.Adam,
        criterion=nn.MSELoss(reduction="sum")
    )

    # agent_trainer.save("./{}".format(args.ckpt), version=1)
