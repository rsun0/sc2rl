
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self):
        
        d1 = 10
        d2 = 32
        d3 = 32
        d4 = 32
        
        self.activation = F.relu

        self.res1 = ResidualBlock([d1,d2,d3,d4])
        self.res2 = ResidualBlock([d1,d2,d3,d4])
        self.res3 = ResidualBlock([d1,d2,d3,d4])
        self.mlp_inputs2d = nn.Sequential(nn.Linear(16, 128), self.activation, nn.Linear(128, 64))
        
        LSTMfilters = 96
        
        self.Conv2DLSTM = ConvLSTM(d4 + 1, LSTMfilters, 3)
        
    def forward(self, x, player_lastaction_in, conv2dlstm_hidden):
        ### x is of shape (N, W, H, D), already pre-processed
        
        
        inputs_3d = self.forward_inputs3d(x)
        inputs_2d = self.forward_inputs2d(player_lastaction_in)
        
        new_inputs = torch.cat([inputs_3d, inputs_2d], 3)
        
        outputs_2d, (x, new_c) = self.Conv2DLSTM(new_inputs)
    
    def forward_inputs3d(self, x):
    
        h1 = self.res1(x)
        h2 = self.res2(h1)
        h3 = self.res3(h2)
        
    def forward_inputs2d(self, p_la):
        ### p_la is of shape (N, D)
        
        x = mlp_inputs2d(p_la)
        x = x.view((-1, 8, 8, 1))
        
        return x
        



class ResidualBlock(nn.Module):

    def __init__(self, dims):
    
        [d1,d2,d3,d4] = dims
        
        self.conv1 = nn.Conv2d(in_channels=d1, 
                                    out_channels=d2,
                                    kernel_size=4,
                                    stride=2)
                                    
        self.conv2 = nn.Conv2d(in_channels=d2,
                                    out_channels=d3,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
                                    
        self.conv3 = nn.Conv2d(in_channels=d3,
                                    out_channels=d4,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
                                    
        self.activation = F.relu

        
    def forward(self, x):
        ### x is of shape (N, W, H, D)
        
        h1 = self.activation(self.conv1(x))
        h2 = self.activation(self.conv2(x)) + h1
        h3 = self.activation(self.conv3(x)) + h2
        
        return h3
        
    
