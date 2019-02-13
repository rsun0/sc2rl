
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from pysc2.lib.features import SCREEN_FEATURES, MINIMAP_FEATURES

class DeepMind2017Net(nn.Module):

    # Assume all shapes are 3d. The input to the forward functions will be 4d stacks.
    def __init__(self, screen_shape, screen_features, minimap_shape, minimap_features, nonspatial_size, nonspatial_features nonspatial_act_shape, spatial_act_shape):
    
        self.device = 
        FILTERS1 = 16
        FILTERS2 = 32
        self.activation = F.relu
        self.latent_size = 50
        
        self.h, self.w, _ = screen_shape
    
        ((h, w, d_cat), (_, _, d_cont)) = screen_shape
        (_, _, d_cat) = minimap_shape
        (h_out, w_out) = nonspatial_act_shape
    
        self.screen_shape = screen_shape
        self.screen_features = screen_features
        self.minimap_shape = minimap_shape
        self.minimap_features = minimap_features
        self.nonspatial_size = nonspatial_size
        self.nonspatial_features = nonspatial_features
        self.nonspatial_act_shape = nonspatial_act_shape
        self.spatial_act_shape = spatial_act_shape
        
        screen_in = torch.zeros(screen_shape).to(self.device)
        minimap_in = torch.zeros(minimap_shape).to(self.device)
        proc_screen, proc_minimap = self.preprocess_input([screen_in, minimap_in])
        
        d1 = proc_screen.shape[3]
        d2 = proc_minimap.shape[3]
        
        self.screen_embed = nn.Sequential(nn.Conv2d(in_channels=d1,
                                                out_channels=self.latent_size,
                                                kernel_size=1,
                                                stride=1,
                                                padding=1),
                                                
                                          self.activation)
                                          
        self.minimap_embed = nn.Sequential(nn.Conv2d(in_channels=d2,
                                                out_channels=self.latent_size,
                                                kernel_size=1,
                                                stride=1,
                                                padding=1),
                                                
                                          self.activation)
        
        
        self.screen_convs = nn.Sequential(nn.Conv2d(in_channels=latent_size,
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
        
        
        self.minimap_convs = nn.Sequential(nn.Conv2d(in_channels=latent_size,
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
        proc_screen, proc_minimap = self.preprocess_input([screen, minimap])
        
        embedded_screen = self.screen_embed(proc_screen)
        screen_out = self.screen_convs(embeded_screen)
        
        embedded_minimap = self.minimap_embed(proc_minimap)
        minimap_out = self.minimap_convs(embedded_minimap)
        
        concat_output = torch.concat([screen_out, minimap_out], dim=3)
        
        return concat_output
        
        
    def forward_spatial(self, features):
        pass
    def forward_nonspatial(self, features):
        pass
    def forward_value(self, features):
        pass
        
    def preprocess_input(self, x):
        '''
        x: (screen_input, minimap_input)
        '''
        (screen_input, minimap_input) = x
        
        preprocessed_screen = self.preprocess_featuremap(screen_input, self.screen_features)
        preprocessed_minimap = self.preprocess_featuremap(minimap_input, self.minimap_features)
        
        
            
    def preprocess_featuremap(self, x, features):
        (w,h,_) = x.shape
        preprocessed_features = np.zeros((w,h,1))
        features_depth = 0
        
        for i in range(len(features)):
            name, scale, featuretype = (features[i].name, features[i].scale, features[i].type)
            
            if (featuretype == FeatureType.Categorical):
                dim = scale
                if (dim == 2):
                    addition = np.expand_dims(x[name], 2)
                else:
                    addition = (np.arange(dim) == (x[name])[...,None]).astype(int)
            else:
                dim = 1
                addition = np.expand_dims(np.log(np.max(x[name], 0) + 1), 2)
                
            if (i == 0):
                preprocessed_features = addition
            else:
                preprocessed_features = np.concatenate([preprocessed_features, addition], 2)
                
        return torch.from_numpy(np.expand_dim(preprocessed_features, 0)).to(self.device)
        

        











