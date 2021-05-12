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
        self.embed_dim = 128

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
        self.encoder_out_dim = 256

        # 2d WxH encoder image # https://arxiv.org/abs/2010.11929
        self.map_encoder = ViT(
            image_size=50,
            patch_size=10,
            num_classes=self.encoder_out_dim,
            dim=128,
            depth=2,
            channels=1,
            heads=3,
            mlp_dim=256,
            dropout=0,
            emb_dropout=0
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
        # 2d patch VIT encoder
        map_out = self.map_encoder(map)
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
        # Feed color ebmeds to agent embeds
        # TODO
        out = self.fc_1(out)
        out = self.activation(out)
        out = self.fc_out(out)

        return F.log_softmax(out, dim=-1)
