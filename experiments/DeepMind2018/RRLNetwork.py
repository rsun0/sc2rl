import torch
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cudnn.benchmarks = True
import numpy as np
import time

from base_agent.Network import BaseNetwork

from base_agent.sc2env_utils import generate_embeddings, multi_embed, valid_args, get_action_args, batch_get_action_args, is_spatial_arg, env_config, processed_feature_dim, full_action_space
from base_agent.net_utils import ConvLSTM, ResnetBlock, SelfAttentionBlock, Downsampler, SpatialUpsampler, Unsqueeze, Squeeze, FastEmbedding, RelationalModule, Flatten

class RRLModel(BaseNetwork):

    """
        net_config: Specifies parameters for network
        device: "cpu" or "cuda:" + str(put int here)
    """
    def __init__(self, net_config, device="cpu", action_space=full_action_space):
        super(RRLModel, self).__init__(net_config, device)
        self.net_config = net_config
        self.device = device
        self.action_space = torch.from_numpy(action_space.reshape(1, -1)).to(device).byte()

        #self.minimap_embeddings, self.screen_embeddings = generate_embeddings(net_config)
        self.state_embeddings = generate_embeddings(net_config)
        self.state_embeddings = nn.ModuleList(self.state_embeddings)
        self.minimap_features = processed_feature_dim(env_config["raw_minimap"], self.state_embeddings[0])
        self.screen_features = processed_feature_dim(env_config["raw_screen"], self.state_embeddings[1])
        self.player_features = env_config["raw_player"]
        self.state_embedding_indices = [
            env_config["minimap_categorical_indices"],
            env_config["screen_categorical_indices"]
        ]
        #self.action_embedding = FastEmbedding(env_config["action_space"], net_config["action_embedding_size"]).to(self.device)
        self.action_embedding = nn.Embedding(env_config["action_space"], net_config["action_embedding_size"]).to(self.device)
        self.down_layers_minimap = Downsampler(self.minimap_features+1, net_config)
        self.down_layers_screen = Downsampler(self.screen_features+2, net_config)

        self.LSTM_in_size = 2*(4*net_config["down_conv_features"]) + net_config["inputs2d_size"] + 2
        self.convLSTM = ConvLSTM(self.LSTM_in_size,
                                    self.net_config['LSTM_hidden_size'])

        self.spatial_upsampler = SpatialUpsampler(net_config, env_config["spatial_action_depth"])

        self.inputs2d_MLP = nn.Sequential(
            nn.Linear(self.player_features + net_config["action_embedding_size"],
                        128),
            nn.ReLU(),
            nn.Linear(128, net_config["inputs2d_size"])
        )
        """
        self.attention_blocks = nn.Sequential(
            SelfAttentionBlock(net_config['LSTM_hidden_size'] + self.LSTM_in_size,
                                net_config['relational_features'],
                                net_config['relational_heads'])
        )
        for i in range(net_config['relational_depth']-1):
            self.attention_blocks.add_module("Block"+str(i+2),
                                                SelfAttentionBlock(
                                                    net_config['relational_heads'] * net_config['relational_features'],
                                                    net_config['relational_features'],
                                                    net_config['relational_heads']
                                                )
                                            )
        """


        self.attention_blocks = nn.Sequential(
            RelationalModule(net_config['LSTM_hidden_size'],
                                net_config['relational_features'],
                                net_config['relational_heads'])
        )
        for i in range(net_config['relational_depth']-1):
            self.attention_blocks.add_module(
                "Block"+str(i+2),
                RelationalModule(net_config['relational_features'],
                                net_config['relational_features'],
                                net_config['relational_heads'],
                                encode=False)
            )

        self.baseline_layers = nn.Sequential(
            nn.Conv2d(net_config['LSTM_hidden_size'], net_config['relational_features'], kernel_size=3, stride=1, padding=1),
            ResnetBlock(net_config['relational_features'], 3),
            ResnetBlock(net_config['relational_features'], 3),
            ResnetBlock(net_config['relational_features'], 3),
            ResnetBlock(net_config['relational_features'], 3)
        )

        self.relational_processor = nn.Sequential(
            nn.MaxPool2d(2),
            Flatten(1, -1),
            nn.Linear(net_config['relational_heads'] * net_config['relational_features'] * 16,
                        512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )


        self.value_MLP = nn.Sequential(
            nn.Linear(512 + net_config["inputs2d_size"], 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.action_MLP = nn.Sequential(
            nn.Linear(512 + net_config["inputs2d_size"], 512),
            nn.ReLU(),
            nn.Linear(512, env_config["action_space"])
        )

        self.arg_MLP = nn.Linear(net_config["inputs2d_size"] + 512 + net_config["action_embedding_size"],
                                    env_config["arg_depth"]*env_config["max_arg_size"])

        """
        self.spatial_out = nn.Sequential(
            nn.Conv2d(
                net_config["channels3"],
                env_config["spatial_action_depth"],
                kernel_size=1,
                padding=0
            )
        )
        """

        # Generates immutable meshgrid to append to inputs3d every time
        x, y = np.meshgrid(np.linspace(-1,1,net_config["inputs3d_width"]), np.linspace(-1,1,net_config["inputs3d_width"]))
        coordinates = np.stack([x,y])
        self.coordinates = torch.from_numpy(coordinates).float().to(self.device).detach().unsqueeze(0)
        print(x.shape, y.shape, coordinates.shape, self.coordinates.shape)
        self.minimap_action = torch.zeros((1, 1, env_config["minimap_width"], env_config["minimap_width"])).float().to(self.device)
        self.screen_action = torch.zeros((1, 2, env_config["screen_width"], env_config["minimap_width"])).float().to(self.device)


        self.arg_depth = env_config["arg_depth"]
        self.arg_size = env_config["max_arg_size"]
        self.spatial_depth = env_config["spatial_action_depth"]
        self.spatial_size = env_config["spatial_action_size"]

        self.valid_args = torch.from_numpy(valid_args).float().to(self.device)
        self.call_count = 0

    """
        minimap: (N, D, H, W)
        screen: (N, D, H, W)
        player: (N, D)
        last_action: (N, D)
        hidden: (N, 2, D, H, W)
        curr_action: (N, D) (curr_action is not None if and only if training)
        choosing: True if sampling an action, False otherwise

        D is a placeholder for feature dimensions.
    """
    def forward(self, minimap, screen, player, avail_actions, last_action, last_spatials, hidden, curr_action=None, choosing=False, unrolling=False, process_inputs=True, inputs2d=None):
        if (choosing):
            assert (curr_action is None)
        if (not choosing and not unrolling):
            assert (curr_action is not None)

        N = len(minimap)

        if choosing:
            minimap = torch.from_numpy(minimap).to(self.device).float()
            screen = torch.from_numpy(screen).to(self.device).float()
            player = torch.from_numpy(player).to(self.device).float()
            last_action = torch.from_numpy(last_action).to(self.device).long()
            hidden = torch.from_numpy(hidden).to(self.device).float()
            avail_actions = torch.from_numpy(avail_actions).to(self.device).byte()

        t1 = time.time()
        inputs = [minimap, screen]

        if process_inputs:
            [minimap, screen] = self.embed_inputs(inputs, self.net_config)
            [minimap, screen] = [minimap.to(self.device), screen.to(self.device)]
            [minimap, screen] = self.concat_spatial([minimap, screen], last_action, last_spatials)

            processed_minimap = self.down_layers_minimap(minimap)
            processed_screen = self.down_layers_screen(screen)
        else:
            processed_minimap = minimap
            processed_screen = screen

        curr_coordinates = self.coordinates.expand((N,) + self.coordinates.shape[1:])

        t2 = time.time()

        inputs3d = torch.cat([processed_minimap, processed_screen, curr_coordinates], dim=1)
        t3 = time.time()
        if process_inputs:
            embedded_last_action = self.action_embedding(last_action).to(self.device)
            nonspatial_concat = torch.cat([player, embedded_last_action], dim=-1)
            inputs2d = self.inputs2d_MLP(nonspatial_concat)

        t4 = time.time()

        expanded_inputs2d = inputs2d.unsqueeze(2).unsqueeze(3).expand(inputs2d.shape + (self.net_config["inputs3d_width"], self.net_config["inputs3d_width"]))
        LSTM_in = torch.cat([inputs3d, expanded_inputs2d], dim=1)

        outputs2d, hidden = self.convLSTM(LSTM_in, hidden)
        #outputs2d = torch.cat([outputs2d, LSTM_in], dim=1)
        t5 = time.time()
        if unrolling:
            #print("Unrolling times\n embedding: %f, down layers: %f, inputs2dmlp: %f, lstm: %f. Total: %f" % (t2-t1,t3-t2,t4-t3,t5-t4,t5-t1))
            return hidden

        relational_spatial = self.attention_blocks(outputs2d)
        #relational_spatial = self.baseline_layers(outputs2d)
        relational_nonspatial = self.relational_processor(relational_spatial)

        t6 = time.time()
        shared_features = torch.cat([inputs2d, relational_nonspatial], dim=-1)
        value = self.value_MLP(shared_features)
        action_logits_in = self.action_MLP(shared_features)
        action_logits_in = action_logits_in.masked_fill((1-avail_actions*self.action_space).bool(), float('-inf'))
        #print("actions: ", torch.max(action_logits_in, dim=-1))
        action_logits = F.softmax(action_logits_in)
        #action_logits = action_logits / torch.sum(action_logits, axis=-1)
        choice = None
        if (choosing):
            action = self.sample_action(action_logits)
            processed_action = torch.from_numpy(np.array(action)).unsqueeze(0).long().to(self.device)
        else:
            action = curr_action
            processed_action = action
            #processed_action = torch.from_numpy(action).long().to(self.device)

        embedded_action = self.action_embedding(processed_action)
        t7 = time.time()
        shared_conditioned = torch.cat([shared_features, embedded_action], dim=-1)
        arg_logit_inputs = self.arg_MLP(shared_conditioned)
        arg_logit_inputs = arg_logit_inputs.reshape((N, self.arg_depth, self.arg_size))
        arg_logits = self.generate_arg_logits(arg_logit_inputs)
        #print("args: ", torch.sum(arg_logit_inputs, dim=-1))

        args = None
        if (choosing):
            args = self.sample_arg(arg_logits, action)

        w = relational_spatial.shape[-1]

        embedded_action = embedded_action.unsqueeze(2).unsqueeze(3).expand(embedded_action.shape + (w,w))
        spatial_input = torch.cat([relational_spatial, embedded_action], dim=1)
        spatial_logits_in = self.spatial_upsampler(spatial_input)
        spatial_logits = self.generate_spatial_logits(spatial_logits_in)
        #print("spatial: ", torch.sum(spatial_logits_in, dim=(-1,-2)))


        spatial = None
        if (choosing):
            spatial = self.sample_spatial(spatial_logits, action)

        choice = [action, args, spatial]
        t8 = time.time()

        #print("full forward times\n embedding: %f, down layers: %f, inputs2dmlp: %f, lstm: %f. relational: %f. value: %f. Selection: %f. Total: %f" % (t2-t1,t3-t2,t4-t3,t5-t4, t6-t5, t7-t6, t8-t7, t8-t1))


        return action_logits, arg_logits, spatial_logits, hidden, value, choice


    def concat_spatial(self, inputs, last_action, last_spatials):
        [minimap, screen] = inputs
        if (len(last_spatials.shape) == 2):
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
            minimap = torch.cat([minimap, minimap_action], dim=1)
            screen = torch.cat([screen, screen_action], dim=1)


        else:
            N = minimap.shape[0]
            minimap_action = self.minimap_action.expand((N,) + self.minimap_action.shape[1:])
            screen_action = self.screen_action.expand((N,) + self.screen_action.shape[1:])
            ids = batch_get_action_args(last_action)
            for i in range(len(ids)):
                if 0 in ids[i]:
                    idx = 2*last_spatials[i,0]
                    screen_action[i,0, idx[0], idx[1]] = 1
                if 1 in ids[i]:
                    idx = 2*last_spatials[i,1]
                    minimap_action[i,0,idx[0],idx[1]] = 1
                if 2 in ids[i]:
                    idx = 2*last_spatials[i,2]
                    screen_action[i,1,idx[0],idx[1]] = 1
            minimap = torch.cat([minimap, minimap_action], dim=1)
            screen = torch.cat([screen, screen_action], dim=1)


        return [minimap, screen]
