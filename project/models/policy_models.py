# Ref https://github.com/lukemelas/EfficientNet-PyTorch
import torch
import torch.nn.functional as F
from torch import nn, Tensor


class PolicyModel(nn.Module):
    def __init__(self, width: int, height: int, action_dim: int = 1):
        super(PolicyModel, self).__init__()

        self.width = width
        self.height = height
        self.embed_dim = 64

        self.fc_1 = nn.Linear(width * height, self.embed_dim)
        self.fc_2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.fc_out = nn.Linear(self.embed_dim, action_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
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
            nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1))
        )

    def forward(self, x):
        return self.block(x)


class ActorPolicyModel(nn.Module):
    def __init__(self, width: int, height: int, action_dim: int):
        super(ActorPolicyModel, self).__init__()

        self.width = width
        self.height = height
        self.encoder_out_dim = 128

        # map features
        self.map_encoder = NaturalBlock()
        self.goal_map_encoder = NaturalBlock()
        self.color_map_encoder = NaturalBlock()

        # 2 features [x,y]
        self.fc_agent_1 = nn.Linear(2, self.encoder_out_dim)
        self.fc_agent_2 = nn.Linear(self.encoder_out_dim, self.encoder_out_dim)

        self.fc_out = nn.Linear(self.encoder_out_dim, action_dim)
        self.activation = nn.ReLU()

    def forward(self, map: Tensor,
                map_goal: Tensor,
                map_colors: Tensor,
                agent_pos: Tensor, ) -> Tensor:
        # see current state
        map_out = self.map_encoder(map)
        map_out = map_out.view(-1, 1, 32 * 4)

        # see end state
        map_goal_out = self.goal_map_encoder(map_goal)
        map_goal_out = map_goal_out.view(-1, 1, 32 * 4)

        # see colors
        map_colors_out = self.color_map_encoder(map_colors)
        map_colors_out = map_colors_out.view(-1, 1, 32 * 4)

        # map passes out
        maps_out = torch.cat((map_out, map_goal_out, map_colors_out))

        # agent pass
        agent_pos_out = self.fc_agent_1(agent_pos)
        agent_pos_out = self.activation(agent_pos_out)
        agent_pos_out = self.fc_agent_2(agent_pos_out)
        agent_pos_out = self.activation(agent_pos_out)

        # out pass
        # combine pos of agents with passes
        out = torch.einsum("ijk,tjk -> tjk", maps_out, agent_pos_out)
        out = self.fc_out(out)

        return F.log_softmax(out, dim=-1)
