import sys
from typing import List

from agents.mappo_trainer import MAPPOTrainer
from environment.action import Action
from environment.env_wrapper import MultiAgentEnvWrapper
from models.policy_models import ActorPolicyModel, CriticPolicyModel

client_name = "46"


def get_server_lines(server_out):
    lines = []
    line = ""
    while not line.startswith("#end"):
        line = server_out.readline()
        lines.append(line)
    return lines


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
    width = 50
    height = 50

    env_wrapper = MultiAgentEnvWrapper({'random': True, 'level_names': level_file_paths_man})
    actor = ActorPolicyModel(width, height, env_wrapper.action_space_n).cuda()
    critic = CriticPolicyModel(width, height).cuda()

    agent_trainer = MAPPOTrainer(
        env_wrapper,
        actor,
        critic
    )

    agent_trainer.save("./ckpt/agent.ckpt")

    agent_trainer.act()
    agent_trainer.env.step()
