import torch
import torch.nn as nn
import torch.nn.Functional as F

from agent import Model

from sc2env_utils import generate_embeddings, multi_embed
from net_utils import ConvLSTM, ResnetBlock, SelfAttentionBlock, Downsampler, SpatialUpsampler, Unsqueeze

class Model(nn.Module, Model):

    """
        net_config = {
            "minimap_features": int, number of features in minimap image
            "screen_features": int, number of features in screen image
            "player": int, number of features in player variable
            "last_action_features": int, number of features in last_action variable
            "down_conv_features": int, numbe features coming out of first convolution
            "up_features": int, number of features going into first upsample
            "up_conv_features": int, number of features coming out of first upsample
            "relational_spatial_depth": int, number of features going into output conv
            "spatial_action_depth": int, max number of spatial arguments for action
            "resnet_features": int, number of features in each convolutional layer
            "resnet_depth": int, number of convolutional layers in resnet block
            "LSTM_in_size": int, number of features in the input to the LSTM
            "LSTM_hidden_size:" int, number of features in the hidden state of LSTM
            "inputs2d_size": int, number of features in inputs2d variable
            "relational_features": int, number of features in each relational block
            "relational_depth": int, number of relational blocks to put in sequence
            "screen_categorical_indices": int array, binary mask. i'th index is 1
                                            iff i'th layer of screen is categorical
            "minimap_categorical_indices": Like screen_categorical_indices.
            "player_categorical_indices": Like screen_categorical_indices.
            "screen_categorical_size": int array, number of screen categories
                                        to embed, corresponds with *_indices
            "minimap_categorical_size": int array, number of mmap categories
                                        to embed, corresponds with *_indices
            "player_categorical_size": int array, number of player categories
                                        to embed, corresponds with *_indices
            "embedding_size": int, number of features output by embeddings

        }
    """
    def __init__(self, net_config):
        self.net_config = net_config

        self.embeddings = generate_embeddings(net_config)
        self.embedding_indices = [
            net_config["minimap_categorical_indices"],
            net_config["screen_categorical_indices"],
            net_config["player_categorical_indices"]
        ]

        self.down_layers_minimap = self.Downsampler(net_config['minimap_features'], net_config)
        self.down_layers_screen = self.Downsampler(net_config['screen_features'], net_config)

        self.convLSTM = ConvLSTM(self.net_config['LSTM_in_size'],
                                    self.net_config['LSTM_hidden_size'])

        self.spatial_upsampler = SpatialUpsampler(net_config)

        self.inputs2d_MLP = nn.Sequential(
            nn.Linear(net_config['player_features'] + net_config['last_action_features'],
                        128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        self.attention_blocks = nn.Sequential(
            SelfAttentionBlock(net_config['inputs2d_size'],
                                net_config['relational_features'])
        )
        for i in range(net_config['relational_depth']-1):
            self.attention_blocks.add_module("Block"+str(i+2),
                                                SelfAttentionBlock(
                                                    net_config['relational_features'],
                                                    net_config['relational_features']
                                                )
                                            )

        self.relational_processor = nn.Sequential(
            nn.MaxPool2d(net_config['inputs3d_width']),
            Unsqueeze(),
            Unsqueeze(),
            nn.Linear(net_config['inputs3d_width'],
                        512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )


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
    def forward(self, minimap, screen, player, last_action, hidden, curr_action=None, choosing=False):
        if (choosing):
            assert (curr_action is None)
        if (not choosing):
            assert (curr_action is not None)

        inputs = [minimap, screen]
        [minimap, screen] = self.embed_inputs(inputs, net_config)

        processed_minimap = self.down_layers_minimap(minimap)
        processed_screen = self.down_layers_screen(screen)
        inputs3d = torch.cat([processed_minimap, processed_screen], dim=1)

        nonspatial_concat = torch.cat([player, last_action], dim=-1)
        inputs2d = self.inputs2d_MLP(nonspatial_concat)

        ### @TODO: LSTM_in concatenates inputs2d and inputs3d
        LSTM_in = None

        outputs2d, hidden = self.convLSTM(LSTM_in, hidden)
        relational_spatial = self.attention_blocks(outputs2d)
        relational_nonspatial = self.relational_processor(relational_spatial)

        shared_features = torch.cat([inputs2d, relational_nonspatial], dim=-1)
        value = self.value_MLP(shared_features)
        action_logits = self.action_MLP(shared_features)

        choice = None
        if (choosing):
            action = self.sample_action(action_logits)
        else:
            action = curr_action

        embedded_action = self.embed_action(action)
        shared_conditioned = torch.cat([shared_features, embedded_action], dim=-1)
        arg_logit_inputs = self.arg_MLP(shared_conditioned)

        arg = None
        if (choosing):
            arg = self.sample_arg(arg_logit_inputs)

        spatial_input = self.spatial_upsampler(relational_spatial)
        ### @TODO: spatial_input appends embedded_action
        spatial_logits = self.spatial_out(spatial_input)

        spatial = None
        if (choosing):
            spatial = self.sample_spatial(spatial_logits)

        choice = [action, arg, spatial]




        return action_logits, arg_logits, spatial_logits, hidden, value, choice


    def embed_inputs(self, inputs, net_config):
        return multi_embed(inputs, self.embeddings, self.embedding_indices)























#############################
