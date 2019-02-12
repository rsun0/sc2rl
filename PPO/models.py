
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class DeepMind2017Net(nn.Module):

    # Assume all shapes are 3d. The input to the forward functions will be 4d stacks.
    def __init__(self, screen_shape, minimap_shape, nonspatial_act_shape, spatial_act_shape):
    
        FILTERS1 = 16
        FILTERS2 = 32
        self.activation = F.relu
    
        (h, w, d1) = screen_shape
        (_, _, d2) = minimap_shape
        (h_out, w_out) = nonspatial_act_shape
    
        self.screen_shape = screen_shape
        self.minimap_shape = minimap_shape
        self.nonspatial_act_shape = nonspatial_act_shape
        self.spatial_act_shape = spatial_act_shape
        
        self.screen_convs == nn.Sequential(nn.Conv2d(in_channels=d1,
                                                out_channels=FILTERS1,
                                                kernel_size=5,
                                                stride=1,
                                                padding=2),
                                                
                                            self.activation,
                                            
                                            nn.Conv2d(in_channels=FILTERS1,
                                                out_channels=FILTERS2,
                                                kernel_size=3
                                                stride=1,
                                                padding=1),
                                                
                                            self.activation)
        
        
        self.minimap_convs == nn.Sequential(nn.Conv2d(in_channels=d2,
                                                out_channels=FILTERS1,
                                                kernel_size=5,
                                                stride=1,
                                                padding=2),
                                                
                                            self.activation,
                                            
                                            nn.Conv2d(in_channels=FILTERS1,
                                                out_channels=FILTERS2,
                                                kernel_size=3
                                                stride=1,
                                                padding=1),
                                                
                                            self.activation)
                                            
    
    
    def forward_features(self, screen, minimap):
        
        
    def forward_spatial(self, features):
        pass
    def forward_nonspatial(self, features):
        pass
    def forward_value(self, features):
        pass
