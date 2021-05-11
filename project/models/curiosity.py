import torch
import torch.nn.functional as F
from torch import nn


# https://pathak22.github.io/noreward-rl/resources/icml17.pdf
class ICMHead(nn.Module):
    def __init__(self, channels=1) -> None:
        super(ICMHead, self).__init__()

        self.conv1 = nn.Conv2d(channels, 32, kernel_size=(8, 8), stride=(4, 4))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
        self.conv3 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1))
        self.dense = nn.Linear(32 * 4, 512)
        self.activation = nn.ReLU()

    def forward(self, state):
        ## TODO FORMAT STATE map agent # no motion blur

        out = self.activation(self.conv1(state))
        out = self.activation(self.conv2(out))
        out = self.activation(self.conv3(out))
        out = out.view(-1, 32 * 4)
        out = self.activation(self.dense(out))
        return out


# Intrinsic Curiosity Model Reward
class IntrinsicCuriosityModule(nn.Module):
    def __init__(self, action_space_n: int, width: int = 64, height: int = 64, max_agents=9) -> None:
        super(IntrinsicCuriosityModule, self).__init__()

        self.head = ICMHead()
        self.state_size = width * height
        self.feature_size = 512
        self.max_agents = max_agents
        self.action_space_n = action_space_n

        self.forward_model = nn.Sequential(
            nn.Linear(self.feature_size + (max_agents * action_space_n), self.feature_size),
            nn.ReLU(),
            nn.Linear(self.feature_size, self.feature_size)
        )

        self.inverse_model = nn.Sequential(
            nn.Linear(self.feature_size * 2, self.feature_size),
            nn.ReLU(),
            nn.Linear(self.feature_size, action_space_n),
            nn.ReLU()
        )

    def forward(self, state, next_state, action):
        phi_t = self.head(state)
        phi_t1 = self.head(next_state)

        action = action.view(-1, self.max_agents * self.action_space_n)
        phi_t1_hat = self.forward_model(torch.cat((phi_t, action), 1))
        a_t_hat = F.softmax(self.inverse_model(torch.cat((phi_t, phi_t1), 1)), dim=-1)
        return a_t_hat, phi_t1_hat, phi_t1, phi_t
