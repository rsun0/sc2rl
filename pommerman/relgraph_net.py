import sys
sys.path.insert(0, "../interface/")

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from agent import Model


class RelGraphNet(nn.Module, Model):
    def __init__(self, board_size, in_channels):
        super().__init__()

        self.pre_prev = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        self.pre_curr = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        combined_dim = 2 * 64 * (board_size // 4) ** 2
        self.pre_combined = nn.Sequential(
            nn.Linear(combined_dim, 1024),
            nn.ReLU(),
        )

        self.ggn_core = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.ggn_post = nn.Sequential(

        )

    def forward(self, state):
        prev_state, curr_state, prev_graph = state
        # TODO convert to pytorch

        processed_prev = self.pre_prev(prev_state)
        processed_prev = torch.flatten(processed_prev, start_dim=1)
        processed_curr = self.pre_curr(curr_state)
        processed_curr = torch.flatten(processed_curr, start_dim=1)
        preprocessed = torch.cat((processed_prev, processed_curr), dim=1)
        preprocessed = self.pre_combined(preprocessed)

        # TODO include prev_graph

if __name__ == '__main__':
    bs = 8
    cs = 23
    i = np.random.random((10, cs, bs, bs)).astype('float32')
    i = torch.from_numpy(i)
    m = RelGraphNet(bs, cs)
    o = m((i, i, i))