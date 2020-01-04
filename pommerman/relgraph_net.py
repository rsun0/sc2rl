import sys
sys.path.insert(0, "../interface/")

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from agent import Model


class GraphGenNet(nn.Module, Model):
    def __init__(self, board_size, in_channels, graph_dim):
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
        self.pre_graph = nn.Sequential(
            nn.Linear(graph_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
        )

        self.ggn_core = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.ggn_post = nn.Sequential(
            nn.Linear(128, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(1024, graph_dim),
        )

    def forward(self, inputs):
        assert len(inputs) == 3
        inputs = list(inputs)
        for i in range(3):
            if isinstance(inputs[i], np.ndarray):
                inputs[i] = torch.from_numpy(inputs[i]).type(torch.FloatTensor)
        prev_state, curr_state, prev_graph = inputs

        prev = self.pre_prev(prev_state)
        prev = torch.flatten(prev, start_dim=1)
        curr = self.pre_curr(curr_state)
        curr = torch.flatten(curr, start_dim=1)
        combined = torch.cat((prev, curr), dim=1)
        combined = self.pre_combined(combined)
        graph = torch.flatten(prev_graph, start_dim=1)
        graph = self.pre_graph(graph)
        preprocessed = torch.cat((combined, graph), dim=1)

        new_graph = self.ggn_core(preprocessed)
        new_graph = self.ggn_post(new_graph)
        return new_graph


if __name__ == '__main__':
    bs = 8
    cs = 23
    graph = np.random.random((10, 2, 66)).astype('float32')
    graph = torch.from_numpy(graph)
    i = np.random.random((10, cs, bs, bs)).astype('float32')
    i = torch.from_numpy(i)
    m = GraphGenNet(bs, cs, 66*2)
    o = m((i, i, graph))
    print(o.shape)
    print(o)