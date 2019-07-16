import torch
import torch.nn as nn
import torch.nn.Functional as F

from agent import Model

from utils import ConvLSTM, ResnetBlock, SelfAttentionBlock, Downsampler, SpatialUpsampler

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
        }
    """
    def __init__(self, net_config):
        self.net_config = net_config

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



















#############################
