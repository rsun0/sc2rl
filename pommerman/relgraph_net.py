import sys
sys.path.insert(0, "../interface/")

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from agent import Model


class RelGraphNet(nn.Module, Model):
    def __init__(self,
        num_agents,
        num_objects,
        init_eps=0.0001,
    ):
        super().__init__()
        self.init_eps = init_eps
        self.graph_dim = (num_agents, num_agents * num_objects)

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

    def reset(self):
        self.prev_graph = np.random.random(self.graph_dim).astype('float32') + self.init_eps
        self.prev_state = None