# Ref https://github.com/lukemelas/EfficientNet-PyTorch
import torch
from torch import nn

from models.SqueezeNet import SqueezeNet


class PolicyModel(nn.Module):
    def __init__(self, width: int, height: int, action_dim: int = 1):
        super(PolicyModel, self).__init__()

        self.width = width
        self.height = height

        self.fc_1 = nn.Linear(width * height, width * height)
        self.fc_2 = nn.Linear(width * height, width * height)
        self.fc_3 = nn.Linear(width * height, height)
        self.fc_out = nn.Linear(height, action_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = x.view(-1, self.motion_blur * self.width * self.height)
        out = self.fc_1(out)
        out = self.activation(out)
        out = self.fc_2(out)
        out = self.activation(out)
        out = self.fc_3(out)
        out = self.activation(out)
        return self.fc_out(out)


class PolicyModelEncoder(nn.Module):
    def __init__(self, width: int, height: int, action_dim: int):
        super(PolicyModelEncoder, self).__init__()

        self.width = width
        self.height = height

        self.scale_down_encoder = SqueezeNet(1, width * height)
        self.embed = nn.Embedding(32, 64)

        self.map_encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=2)
        self.map_encoder = nn.TransformerEncoder(self.map_encoder_layer, num_layers=3)

        self.fc_agent_1 = nn.Linear(2, width * height)
        self.fc_agent_2 = nn.Linear(width * height, width * height)

        self.fc_1 = nn.Linear(width * height, width)
        self.fc_out = nn.Linear(width, action_dim)
        self.activation = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, map, agent_map, map_mask=None):
        map_out = self.embed(map)
        print(map_out.shape)

        if map_mask is not None:
            map_mask = map_mask.view(1, self.width * self.height)

        map_out = map.view(1, -1, self.width * self.height)
        map_out = map_out.permute(1, 0, 2)
        print(map_out.shape)
        print(map_mask.shape)
        map_out = self.map_encoder(map_out, src_key_padding_mask=map_mask)
        map_out = map_out.view(self.width, self.height)

        agent_map_out = self.fc_agent_1(agent_map)
        self.activation(agent_map_out)
        agent_map_out = self.fc_agent_2(agent_map_out)
        agent_map_out = agent_map_out.view(-1, self.width, self.height)

        # Feed attention weights to agents
        out = torch.einsum("jk,tjk -> tjk", map_out, agent_map_out)
        out = out.view(-1, self.width * self.height)
        out = self.fc_1(out)
        out = self.activation(out)
        out = self.fc_out(out)
        out = self.activation(out)

        return self.log_softmax(out)
