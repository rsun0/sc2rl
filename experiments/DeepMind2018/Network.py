import torch
import torch.nn as nn
import torch.nn.Functional as F

from agent import Model

from utils import ConvLSTM, ResnetBlock, SelfAttentionBlock, Downsampler, SpatialUpsampler

class Model(nn.Module, Model):

    """
        net_config = {

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
