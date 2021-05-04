from torch import nn
# Ref https://github.com/lukemelas/EfficientNet-PyTorch
from efficientnet_pytorch import EfficientNet

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

        self.scale_down_encoder_eff = EfficientNet.from_name('efficientnet-b0', in_channels=1, num_classes=width)
        self.map_encoder_layer = nn.TransformerEncoderLayer(d_model=width, nhead=2)
        self.map_encoder = nn.TransformerEncoder(self.map_encoder_layer, num_layers=3)

        self.agent_encoder_layer = nn.TransformerEncoderLayer(d_model=width, nhead=2)
        self.agent_encoder = nn.TransformerEncoder(self.map_encoder_layer, num_layers=3)

        self.fc_out = nn.Linear(width * 8, action_dim)
        self.activation = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=0)

    def forward(self, map, agent_map, map_mask=None):

        map_out = self.scale_down_encoder_eff(map)
        map_out = map_out.unsqueeze(0).permute(1, 0, 2)
        map_out = self.map_encoder(map_out, map_mask)

        agent_map_out = self.agent_encoder(agent_map)

        # TODO
        out = self.fc_out(map_out)
        return self.log_softmax(out)
