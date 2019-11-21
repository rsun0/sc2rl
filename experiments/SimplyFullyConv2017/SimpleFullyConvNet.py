import torch
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cudnn.benchmarks = True
import numpy as np
import time

from base_agent.Network import BaseNetwork
import matplotlib.pyplot as plt

from base_agent.sc2env_utils import generate_embeddings, multi_embed, valid_args, get_action_args, batch_get_action_args, is_spatial_arg, env_config, processed_feature_dim, full_action_space
from base_agent.net_utils import ConvLSTM, ResnetBlock, SelfAttentionBlock, Downsampler, SpatialUpsampler, Unsqueeze, Squeeze, FastEmbedding, RelationalModule, Flatten


class ConvNet(BaseNetwork):

    def __init__(self, net_config, device="cpu", action_space=full_action_space):
        super(BaseNetwork, self).__init__(net_config, device)
        self.net_config = net_config
        self.device = device
        self.action_space = torch.from_numpy(action_space.reshape((1,-1)).to(device).byte())
        self.hist_depth = net_config["history_size"]

        self.spatial_depth = env_config["spatial_action_depth"]

        self.state_embeddings = generate_embedding(net_config)
        self.state_embeddings = nn.ModuleList(self.state_embeddings)
        self.minimap_features = processed_feature_dim(env_config["raw_minimap"], self.state_embeddings[0])
        self.screen_features = processed_feature_dim(env_config["raw_screen"], self.state_embeddings[1])
        self.player_features = env_config["raw_player"]
        self.state_embedding_indices = [
            env_config["minimap_categorical_indices"],
            env_config["screen_categorical_indices"]
        ]

        self.action_embedding = nn.Embedding(env_config["action_space"])

        self.screen_layers = nn.Sequential(
            nn.Conv2d(self.hist_depth * (self.screen_features+1), 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.minimap_layers = nn.Sequential(
            nn.Conv2d(self.hist_depth * (self.minimap_features+1), 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.spatial_conv = nn.Conv2d(128, self.spatial_depth, kernel_size=1, stride=1, padding=0)
