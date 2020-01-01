import sys
sys.path.insert(0, "../interface/")

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from agent import Model


class RelGraphNet(nn.Module, Model):
    def __init__(self):
        super().__init__()

        self.ggn_core = nn.Sequential(
            nn.Linear(, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.ggn_post = nn.Sequential(

        )

    def forward(self, state):
        prev_state, curr_state, prev_graph = state