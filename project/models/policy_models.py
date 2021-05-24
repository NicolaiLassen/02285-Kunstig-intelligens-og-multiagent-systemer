from typing import Tuple

import torch
from torch import nn, Tensor


class CriticPolicyModel(nn.Module):
    def __init__(self, width: int, height: int, action_dim: int = 1):
        super(CriticPolicyModel, self).__init__()

        self.width = width
        self.height = height
        self.embed_dim = 64

        self.fc_1 = nn.Linear(width * height, self.embed_dim)
        self.fc_2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.fc_out = nn.Linear(self.embed_dim, action_dim)
        self.activation = nn.ReLU()

    def forward(self, x, action):
        state_action = torch.cat([x, action], 1)
        x = x.view(-1, self.width * self.height)
        out = self.fc_1(x)
        out = self.activation(out)
        out = self.fc_2(out)
        out = self.activation(out)
        return self.fc_out(out)


class NaturalBlock(nn.Module):
    def __init__(self):
        super(NaturalBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(8, 8), stride=(4, 4)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1)),
        )

    def forward(self, x):
        return self.block(x)


class ActorPolicyModel(nn.Module):
    def __init__(self, width: int = 50, height: int = 50, action_dim: int = 29):
        super(ActorPolicyModel, self).__init__()

        self.width = width
        self.height = height
        self.embed_dim = 128

        # map features
        self.map_encoder = NaturalBlock()
        self.goal_map_encoder = NaturalBlock()
        self.color_map_encoder = NaturalBlock()

        # map passes
        self.fc_map_passes = nn.Linear(3 * self.embed_dim, self.embed_dim)

        # agent features
        self.agent_embed = nn.Linear(2, self.embed_dim)

        self.fc_passes = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.fc_out = nn.Linear(self.embed_dim, action_dim)
        self.activation = nn.ReLU()

    def forward(self, state: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        map_x = state[0].view(-1, 1, self.width, self.height).float().cuda()
        map_goal_x = state[1].view(-1, 1, self.width, self.height).float().cuda()
        map_colors_x = state[2].view(-1, 1, self.width, self.height).float().cuda()
        agent_x = state[3].float().view(-1, 2).cuda()

        # see state
        map_out = self.map_encoder(map_x)
        map_out = map_out.view(-1, 1, 64 * 2)

        # see end state
        map_goal_out = self.goal_map_encoder(map_goal_x)
        map_goal_out = map_goal_out.view(-1, 1, 64 * 2)

        # see colors
        map_colors_out = self.color_map_encoder(map_colors_x)
        map_colors_out = map_colors_out.view(-1, 1, 64 * 2)

        # map passes out
        maps_out = torch.cat((map_out, map_goal_out, map_colors_out)).view(-1, 3 * self.embed_dim)
        # embed map passes out
        map_passes_out = self.activation(self.fc_map_passes(maps_out))

        # combine pos of agents with passes
        # solving the where are my agent problem
        agent_out = self.activation(self.agent_embed(agent_x))
        all_passes_out = torch.cat((map_passes_out, agent_out)).view(-1, self.embed_dim * 2)

        # out the beast of all passes
        out = self.activation(self.fc_passes(all_passes_out))
        return self.fc_out(out)
