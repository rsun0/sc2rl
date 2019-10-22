import sys
sys.path.insert(0, "../interface/")

from agent import Model
from torch import nn

class MCTSPolicyNet(nn.module, Model):
    def __init__(self, board_size=8, in_channels=13, num_scalars=6):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(13, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        width = board_size ** 2 + num_scalars
        self.mlp = nn.Sequential(
            nn.Linear(width, width),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Linear(width, 1)
        )

    def forward(self, state):
        # TODO separate image and scalars from state
        pass