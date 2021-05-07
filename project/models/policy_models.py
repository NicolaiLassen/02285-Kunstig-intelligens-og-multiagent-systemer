# Ref https://github.com/lukemelas/EfficientNet-PyTorch
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from vit_pytorch import ViT


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

    def forward(self, map: Tensor) -> Tensor:
        out = map.view(-1, self.width * self.height)
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
        self.encoder_out_dim = 512

        # 1D S encoder
        # self.d_mode = 16
        # self.fc_map_1 = nn.Linear(width * height, self.d_model * width)
        # self.map_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=2)
        # self.map_encoder = nn.TransformerEncoder(self.map_encoder_layer, num_layers=2)

        # 2d WxH encoder image # https://arxiv.org/abs/2010.11929
        self.map_encoder = ViT(
            image_size=width,
            patch_size=25,
            num_classes=self.encoder_out_dim,
            dim=512,
            depth=4,
            channels=1,
            heads=4,
            mlp_dim=1024,
            dropout=0.1,
            emb_dropout=0.1
        )

        # 2 features [x,y]
        self.fc_agent_1 = nn.Linear(2, self.encoder_out_dim)
        self.fc_agent_2 = nn.Linear(self.encoder_out_dim, self.encoder_out_dim)

        self.fc_1 = nn.Linear(self.encoder_out_dim, width)
        self.fc_out = nn.Linear(width, action_dim)
        self.activation = nn.ReLU()

    def forward(self, map: Tensor,
                agent_map: Tensor,
                color_map: Tensor = None,
                map_mask: Tensor = None) -> Tensor:
        # map pass 1D encoder
        # Embed dims d_model
        # map_out = map.view(-1, self.width * self.height)
        # map_out = self.fc_map_1(map_out)
        # map_out = self.activation(map_out)
        # map_out = map_out.view(self.width, -1, self.d_model)
        # map_out = self.map_encoder(map_out, src_key_padding_mask=None).transpose(0, 1)

        # 2d patch VIT encoder
        map_out = map.view(-1, 1, self.width, self.height)
        map_out = self.map_encoder(map_out)
        map_out = map_out.unsqueeze(1)

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
        out = self.fc_1(out)
        out = self.activation(out)
        out = self.fc_out(out)
        return F.log_softmax(out, dim=-1)
