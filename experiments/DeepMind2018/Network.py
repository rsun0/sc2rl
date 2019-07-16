import torch
import torch.nn as nn
import torch.nn.Functional as F

from agent import Model

from utils import ConvLSTM, ResnetBlock, SelfAttentionBlock

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

        self.down_layers_minimap = self.create_downsample(net_config['minimap_features'], net_config)
        self.down_layers_screen = self.create_downsample(net_config['screen_features'], net_config)

        self.convLSTM = ConvLSTM(self.net_config['LSTM_in_size'],
                                    self.net_config['LSTM_hidden_size'])


    def create_downsample(net_config, input_features):
        down_layers = nn.Sequential(
            nn.Conv2d(net_config[input_features]],
                        net_config['down_conv_features'],
                        kernel_size=(4,4),
                        stride=2,
                        padding=2),
            nn.ReLU(),

            ResnetBlock(self.net_config['down_conv_features'],
                            self.net_config['resnet_depth']),
            ResnetBlock(self.net_config['down_conv_features'],
                            self.net_config['resnet_depth']),

            nn.Conv2d(net_config['down_conv_features'],
                        2*net_config['down_conv_features'],
                        kernel_size=(4,4),
                        stride=2,
                        padding=2),
            nn.ReLU(),

            ResnetBlock(2*self.net_config['down_conv_features'],
                            self.net_config['resnet_depth']),
            ResnetBlock(2*self.net_config['down_conv_features'],
                            self.net_config['resnet_depth']),

            nn.Conv2d(2*net_config['down_conv_features'],
                        4*net_config['down_conv_features'],
                        kernel_size=(4,4),
                        stride=2,
                        padding=2),
            nn.ReLU(),

            ResnetBlock(4*self.net_config['down_conv_features'],
                            self.net_config['resnet_depth']),
            ResnetBlock(4*self.net_config['down_conv_features'],
                            self.net_config['resnet_depth']),

        )
    return down_layers
