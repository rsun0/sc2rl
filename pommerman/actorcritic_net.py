import sys
sys.path.insert(0, "../interface/")

import torch
from torch import nn

from agent import Model


class ResBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super().__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.main = nn.Sequential(
            nn.Conv2d(inplanes, planes, 3, padding=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, 3, padding=1),
            nn.BatchNorm2d(planes),
        )

    def forward(self, x):
        identity = x
        out = self.main(x)
        out += identity
        out = torch.relu_(out)
        return out


class ActorCriticNet(nn.Module, Model):
    def __init__(self,
            board_size=8,
            in_channels=13,
            num_blocks=4,
            channels=32,
            num_actions=6):
        super().__init__()

        # Convolutions
        convs = [
            nn.Conv2d(in_channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        ]
        for i in range(num_blocks):
            convs.append(ResBlock(channels, channels))
        convs.extend([
            nn.Conv2d(channels, 4, 1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
        ])
        self.shared_convs = nn.Sequential(*convs)
        
        fc_h = 4 * board_size ** 2
        self.shared_fcs = nn.Sequential(
            nn.Linear(fc_h, fc_h),
            nn.BatchNorm1d(fc_h),
            nn.ReLU()
        )

        self.actor = nn.Sequential(
            nn.Linear(fc_h, fc_h),
            nn.BatchNorm1d(fc_h),
            nn.ReLU(),
            nn.Linear(fc_h, num_actions)
        )

        self.critic = nn.Sequential(
            nn.Linear(fc_h, fc_h),
            nn.BatchNorm1d(fc_h),
            nn.ReLU(),
            nn.Linear(fc_h, 1)
        )

        self.actor_criterion = nn.CrossEntropyLoss()
        self.critic_criterion = nn.MSELoss()

    def forward(self, state):
        x = torch.from_numpy(state).type(torch.FloatTensor)
        x = self.shared_convs(x)
        x = torch.flatten(x, start_dim=1)
        x = self.shared_fcs(x)
        policy_scores = self.actor(x)
        vals = self.critic(x)
        return policy_scores, vals