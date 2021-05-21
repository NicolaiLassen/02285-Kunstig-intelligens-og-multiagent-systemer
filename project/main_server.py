import sys
# import ray
# from ray.rllib.agents.ppo import ppo
# from ray.tune import register_env
from typing import List

from environment.action import Action
from environment.env_wrapper import EnvWrapper

client_name = "46"


def get_server_lines(server_out):
    level_lines = []
    line = ""
    while not line.startswith("#end"):
        line = server_out.readline()
        level_lines.append(line)
    return level_lines


def send_plan(server_out, plan: List[List[Action]]):
    # Print plan to server.
    if plan is None:
        print('Unable to solve level.', file=sys.stderr, flush=True)
        sys.exit(0)
    else:
        print('Found solution of length {}.'.format(len(plan)), file=sys.stderr, flush=True)
        for joint_action in plan:
            print("|".join(a.name_ for a in joint_action), flush=True)
            # We must read the server's response to not fill up the stdin buffer and block the server.
            response = server_out.readline()


def get_server_out():
    # Send client name to server.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding='ASCII')
    print(client_name, flush=True)

    server_out = sys.stdin
    if hasattr(server_out, "reconfigure"):
        server_out.reconfigure(encoding='ASCII')
    return server_out


if __name__ == '__main__':
    # BEFORE SERVER
    # ray.init(include_dashboard=False)

    server_out = get_server_out()
    level_lines = get_server_lines(server_out)

    env = EnvWrapper({'level_lines': level_lines})
    env_name = "multi_agent_env"


    def env_creator(_):
        return env


    register_env(env_name, env_creator)

    agent = ppo.PPOTrainer(env='multi_agent_env',
                           config={
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

    agent.restore('./ckpt/cevents.out.tfevents.1621554386.n-62-20-3')
    final_actions = []
    s1 = env.reset()
    r1 = None
    a1 = None
    for i in range(10):
        s = s1
        actions = agent.compute_actions(s, prev_reward=r1, prev_action=a1)
        s1, r, d, _ = env.step(actions)
        print(s1)
        if (s[0][0] == s1[0][0]).all():
            continue
        actions_s1 = actions
        final_actions.append(actions)
        print(r)
        if d['__all__']:
            print(d)
            break

    send_plan(server_out, final_actions)
