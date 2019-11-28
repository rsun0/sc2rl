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
        super(ConvNet, self).__init__(net_config, device)
        self.net_config = net_config
        self.device = device
        self.action_space = torch.from_numpy(action_space.reshape((1,-1))).to(device).byte()
        self.hist_depth = net_config["history_size"]

        self.spatial_depth = env_config["spatial_action_depth"]
        self.spatial_action_size = env_config["spatial_action_size"]

        self.state_embeddings = generate_embeddings(net_config)
        self.state_embeddings = nn.ModuleList(self.state_embeddings)
        self.minimap_features = processed_feature_dim(env_config["raw_minimap"], self.state_embeddings[0])
        self.screen_features = processed_feature_dim(env_config["raw_screen"], self.state_embeddings[1])
        self.player_features = env_config["raw_player"]
        self.state_embedding_indices = [
            env_config["minimap_categorical_indices"],
            env_config["screen_categorical_indices"]
        ]

        self.action_embedding = nn.Embedding(env_config["action_space"], net_config["action_embedding_size"])

        self.screen_layers = nn.Sequential(
            nn.Conv2d(self.hist_depth * (self.screen_features+2), 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.minimap_layers = nn.Sequential(
            nn.Conv2d(self.hist_depth * (self.minimap_features+1), 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.spatial_policy_layer = nn.Conv2d(64 + env_config["raw_player"], self.spatial_depth, kernel_size=1, stride=1, padding=0)

        self.fc = nn.Sequential(
            nn.Linear((64)*(self.spatial_action_size ** 2)+env_config["raw_player"], 256),
            nn.ReLU(),
        )

        self.LSTM_in_size = 2*(2*net_config["down_conv_features"]) + net_config["inputs2d_size"] + 2

        self.convLSTM = ConvLSTM(self.LSTM_in_size,
                                    self.net_config['LSTM_hidden_size'])

        self.value_layer = nn.Linear(256, 1)
        self.base_policy_layer = nn.Linear(256, env_config["action_space"])
        self.arg_policy_layer = nn.Linear(256, env_config["arg_depth"]*env_config["max_arg_size"])

        x, y = np.meshgrid(np.linspace(-1,1,net_config["inputs3d_width"]), np.linspace(-1,1,net_config["inputs3d_width"]))
        coordinates = np.stack([x,y])
        self.coordinates = torch.from_numpy(coordinates).float().to(self.device).detach().unsqueeze(0)
        self.minimap_action = torch.zeros((1, 1, env_config["minimap_width"], env_config["minimap_width"])).float().to(self.device)
        self.screen_action = torch.zeros((1, 2, env_config["screen_width"], env_config["minimap_width"])).float().to(self.device)

        self.valid_args = torch.from_numpy(valid_args).float().to(self.device)

        self.arg_depth = env_config["arg_depth"]
        self.arg_size = env_config["max_arg_size"]
        self.spatial_depth = env_config["spatial_action_depth"]
        self.spatial_size = env_config["spatial_action_size"]

    def forward(self, minimap, screen, player, avail_actions, last_action, last_spatials, hidden, curr_action=None, choosing=False, unrolling=False, process_inputs=True, inputs2d=None, format_inputs=True):

        N = len(minimap)

        if (choosing):
            assert(curr_action is None)
        if (not choosing and not unrolling):
            assert(curr_action is not None)

        if choosing:
            minimap = torch.from_numpy(minimap).to(self.device).float()
            screen = torch.from_numpy(screen).to(self.device).float()
            player = torch.from_numpy(player).to(self.device).float()
            last_action = torch.from_numpy(last_action).to(self.device).float()
            hidden = torch.from_numpy(hidden).to(self.device).float()
            avail_actions = torch.from_numpy(avail_actions).to(self.device).byte()
            last_spatials = torch.from_numpy(last_spatials).to(self.device).long()

        if process_inputs:
            if format_inputs:
                # process_states(self, minimaps, screens, players, last_actions, last_spatial_actions, sequential=True, embeddings_only=False)
                #[minimap, screen] = self.embed_inputs(inputs)
                #[minimap, screen] = [minimap.to(self.device), screen.to(self.device)]
                #[minimap, screen] = self.concat_spatial([minimap, screen], last_action, last_spatials)
                [minimap, screen, _] = self.process_states(minimap, screen, player, last_action, last_spatials, sequential=False, embeddings_only=True)

        processed_minimap = self.minimap_layers(minimap)
        processed_screen = self.screen_layers(screen)
        processed_spatial = torch.cat([processed_minimap, processed_screen], dim=1)

        expanded_player = player.unsqueeze(2).unsqueeze(3).expand(player.shape + (self.spatial_action_size, self.spatial_action_size))
        outputs2d = torch.cat([processed_spatial, expanded_player], dim=1)
        outputs1d = torch.cat([processed_spatial.flatten(1,-1), player], dim=1)

        fc_out = self.fc(outputs1d)

        value = self.value_layer(fc_out)
        base_policy = self.base_policy_layer(fc_out)
        arg_policy = self.arg_policy_layer(fc_out)
        arg_policy = arg_policy.reshape((N, self.arg_depth, self.arg_size))
        spatial_policy = self.spatial_policy_layer(outputs2d)

        action_logits = F.softmax(base_policy.masked_fill((1-avail_actions*self.action_space).bool(), float('-inf')))
        arg_logits = self.generate_arg_logits(arg_policy)
        spatial_logits = self.generate_spatial_logits(spatial_policy)

        action, args, spatial = None, None, None
        if (choosing):
            action = self.sample_action(action_logits)
            args = self.sample_arg(arg_logits, action)
            spatial = self.sample_spatial(spatial_logits, action)

        choice = [action, args, spatial]

        return action_logits, arg_logits, spatial_logits, hidden, value, choice


    def concat_spatial(self, inputs, last_action, last_spatials):
        [minimap, screen] = inputs
        N = minimap.shape[0]
        """
        if (N == 1):
            minimap_action = self.minimap_action.clone()
            screen_action = self.screen_action.clone()
            ids = get_action_args(last_action)
            if 0 in ids:
                idx = 2*last_spatials[0]
                screen_action[:,0,idx[0], idx[1]] = 1
            if 1 in ids:
                idx = 2*last_spatials[1]
                minimap_action[:,0,idx[0],idx[1]] = 1
            if 2 in ids:
                idx = 2*last_spatials[2]
                screen_action[:,1,idx[0],idx[1]] = 1
            print(minimap.shape, minimap_action.shape)
            minimap = torch.cat([minimap, minimap_action], dim=1)
            screen = torch.cat([screen, screen_action], dim=1)


        else:
        """
        minimap_action = self.minimap_action.expand((N,) + self.minimap_action.shape[1:])
        screen_action = self.screen_action.expand((N,) + self.screen_action.shape[1:])
        #ids = batch_get_action_args(last_action)
        ### @TODO: Finish this garbage
        valid_spatial = last_spatials[:,:,0] >= 0
        indices = valid_spatial.nonzero()

        batch_inds = indices[:,0]
        #hist_inds = indices[:,1]
        arg_inds = indices[:,1]

        for i in range(len(batch_inds)):
            b = batch_inds[i]
            a = arg_inds[i]
            idx = 2*last_spatials[b,a]
            arr = [screen_action, minimap_action][a==1]
            ind = a // 2
            arr[b,ind,idx[0],idx[1]] = 1

            """
            if 0 in ids[i]:
                idx = 2*last_spatials[i,0]
                print(idx, last_spatials.shape)
                screen_action[i,0, idx[0], idx[1]] = 1
            if 1 in ids[i]:
                idx = 2*last_spatials[i,1]
                print(idx, last_spatials.shape)
                minimap_action[i,0,idx[0],idx[1]] = 1
            if 2 in ids[i]:
                idx = 2*last_spatials[i,2]
                print(idx, last_spatials.shape)
                screen_action[i,1,idx[0],idx[1]] = 1
            """

        minimap = torch.cat([minimap, minimap_action], dim=1)
        screen = torch.cat([screen, screen_action], dim=1)


        return [minimap, screen]







#
