from torch import nn


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
