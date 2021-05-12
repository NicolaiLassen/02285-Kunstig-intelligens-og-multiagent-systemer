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


class ActorPolicyModel(nn.Module):
    def __init__(self, width: int, height: int, action_dim: int):
        super(ActorPolicyModel, self).__init__()

        self.width = width
        self.height = height
        self.encoder_out_dim = 128

        # map features
        self.map_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(8, 8), stride=(4, 4)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1))
        )

        self.goal_map_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(8, 8), stride=(4, 4)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1))
        )

        self.color_map_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(8, 8), stride=(4, 4)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1))
        )

        # 2 features [x,y]
        self.fc_agent_1 = nn.Linear(2, self.encoder_out_dim)
        self.fc_agent_2 = nn.Linear(self.encoder_out_dim, self.encoder_out_dim)

        self.fc_1 = nn.Linear(self.encoder_out_dim, width)
        self.fc_out = nn.Linear(width, action_dim)
        self.activation = nn.ReLU()

    def forward(self, map: Tensor,
                goal_map: Tensor,
                color_map: Tensor,
                agent_pos: Tensor,) -> Tensor:
        map_out = self.map_encoder(map)
        map_out = map_out.view(-1, 1, 32 * 4)

        map_goal_out = self.goal_map_encoder(goal_map)
        map_goal_out = map_goal_out.view(-1, 1, 32 * 4)

        map_color_out = self.color_map_encoder(color_map)
        map_color_out = map_color_out.view(-1, 1, 32 * 4)

        # agent pass
        agent_pos_out = self.fc_agent_1(agent_pos)
        agent_pos_out = self.activation(agent_pos_out)
        agent_pos_out = self.fc_agent_2(agent_pos_out)
        agent_pos_out = self.activation(agent_pos_out)

        # out pass
        # Feed attention weights to agent embeds
        maps_out = torch.cat((map_out, map_goal_out, map_color_out))
        out = torch.einsum("ijk,tjk -> tjk", maps_out, agent_pos_out)

        out = self.fc_1(out)
        out = self.activation(out)
        out = self.fc_out(out)

        return F.log_softmax(out, dim=-1)
