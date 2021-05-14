import os

import torch

from agents.ppo_agent import PPOAgent
from environment.env_wrapper import EnvWrapper
from models.curiosity import IntrinsicCuriosityModule
from models.policy_models import ActorPolicyModel, PolicyModel


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


if __name__ == '__main__':
    width = 50
    height = 50
    create_dir("./ckpt")

    lr_actor = 3e-4
    lr_critic = 1e-3
    lr_icm = 1e-3

    env = EnvWrapper()
    actor = ActorPolicyModel(width, height, env.action_space_n).cuda()
    critic = PolicyModel(width, height).cuda()
    icm = IntrinsicCuriosityModule(env.action_space_n).cuda()

    optimizer = torch.optim.Adam([
        {'params': actor.parameters(), 'lr': lr_actor},
        {'params': icm.parameters(), 'lr': lr_icm},
        {'params': critic.parameters(), 'lr': lr_critic}
    ])

    agent = PPOAgent(env, actor, critic, icm, optimizer)
    agent.train(1000, 200000000)
