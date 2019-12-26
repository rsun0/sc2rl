import sys
sys.path.insert(0, "../interface/")

import torch
from torch import nn

from agent import Model

class MCTSPolicyNet(nn.Module, Model):
    def __init__(self, board_size=8, in_channels=13, num_actions=6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        width = board_size ** 2
        self.mlp = nn.Sequential(
            nn.Linear(width, width),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Linear(width, num_actions)
        )

    def forward(self, state):
        x = torch.from_numpy(state).type(torch.FloatTensor)
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.mlp(x)
        return x