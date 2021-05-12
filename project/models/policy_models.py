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
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(8, 8), stride=(4, 4))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
        self.conv3 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1))
        self.activation = nn.ReLU()

        # 2 features [x,y]
        self.fc_agent_1 = nn.Linear(2, self.encoder_out_dim)
        self.fc_agent_2 = nn.Linear(self.encoder_out_dim, self.encoder_out_dim)

        self.fc_1 = nn.Linear(self.encoder_out_dim, width)
        self.fc_out = nn.Linear(width, action_dim)
        self.activation = nn.ReLU()

    def forward(self, map: Tensor,
                agent_map: Tensor,
                color_map: Tensor = None) -> Tensor:

        map_out = self.activation(self.conv1(map))
        map_out = self.activation(self.conv2(map_out))
        map_out = self.activation(self.conv3(map_out))
        map_out = map_out.view(-1, 1, 32 * 4)

        # agent pass
        agent_map_out = self.fc_agent_1(agent_map)
        agent_map_out = self.activation(agent_map_out)
        agent_map_out = self.fc_agent_2(agent_map_out)
        agent_map_out = self.activation(agent_map_out)

        # color pass
        # TODO

        # out pass
        # Feed attention weights to agent embeds
        out = torch.einsum("ijk,tjk -> tjk", map_out, agent_map_out)
        # Feed color ebmeds to agent embeds
        # TODO
        out = self.fc_1(out)
        out = self.activation(out)
        out = self.fc_out(out)

        return F.log_softmax(out, dim=-1)
