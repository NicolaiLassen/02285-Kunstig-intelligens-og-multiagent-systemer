import torch

from agents.ppo_agent import PPOAgent
from environment.env_wrapper import EnvWrapper
from environment.level_loader import load_level
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


if __name__ == '__main__':
    initial_state, goal_state = load_level(1)

    width = 50
    height = 50

    lr_actor = 3e-4
    lr_critic = 1e-3
    lr_icm = 1e-3

    # print(initial_state, file=sys.stderr, flush=True)
    #
    env = EnvWrapper(
        action_space_n=29,
        initial_state=initial_state,
        goal_state=goal_state,
    )
    # print(env)
    #
    # for a in [Action.PushSS, Action.PushSS, Action.PushSS, Action.MoveN, Action.MoveN, Action.MoveN, Action.MoveE,
    #           Action.MoveE, Action.MoveE]:
    #     s1, r, d = env.step([a])
    #     print(env)
    #     print(d)

    actor = ActorPolicyModel(width, height, env.action_space_n).cuda()
    critic = PolicyModel(width, height).cuda()
    icm = IntrinsicCuriosityModule(env.action_space_n).cuda()

    optimizer = torch.optim.Adam([
        {'params': actor.parameters(), 'lr': lr_actor},
        {'params': icm.parameters(), 'lr': lr_icm},
        {'params': critic.parameters(), 'lr': lr_critic}
    ])

    print(get_n_params(actor))
    print(get_n_params(critic))
    agent = PPOAgent(env, actor, critic, icm, optimizer)
    agent.train(2000, 10000)
