
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from config import *
import numpy as np
import time
import bisect
import random

class DeepMind2017Net(nn.Module):

    # Assume all shapes are 3d. The input to the forward functions will be 4d stacks.
    def __init__(self, nonspatial_act_size, spatial_act_size, device):
        super(DeepMind2017Net, self).__init__()
        self.device = device
    
        self.config = DeepMind2017Config
        FILTERS1 = self.config.FILTERS1
        FILTERS2 = self.config.FILTERS2
        FC1SIZE = self.config.FC1SIZE
        FC2SIZE = self.config.FC2SIZE
        self.latent_size = self.config.latent_size
        
        self.where_yes = torch.ones(1).to(self.device)
        self.where_no = torch.zeros(1).to(self.device)
        
        _, self.h, self.w = self.config.screen_shape
    
        self.screen_shape = self.config.screen_shape
        self.minimap_shape = self.config.minimap_shape
        self.nonspatial_size = self.config.nonspatial_size
        self.nonspatial_act_size = nonspatial_act_size
        self.spatial_act_size = spatial_act_size
        
        
        self.d1 = self.screen_shape[0]
        self.d2 = self.minimap_shape[0]
        self.d3 = self.nonspatial_size
        
        self.screen_embed = nn.Sequential(nn.Conv2d(in_channels=self.d1,
                                                out_channels=self.latent_size,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0),
                                                
                                            nn.Tanh())
                                          
        self.minimap_embed = nn.Sequential(nn.Conv2d(in_channels=self.d2,
                                                out_channels=self.latent_size,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0),
                                                
                                            nn.Tanh())
                                          
        
        self.screen_convs = nn.Sequential(nn.Conv2d(in_channels=self.latent_size,
                                                out_channels=FILTERS1,
                                                kernel_size=5,
                                                stride=1,
                                                padding=2),
                                                
                                            nn.ReLU(),
                                            
                                            nn.Conv2d(in_channels=FILTERS1,
                                                out_channels=FILTERS2,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1),
                                                
                                            nn.ReLU())
        
        
        self.minimap_convs = nn.Sequential(nn.Conv2d(in_channels=self.latent_size,
                                                out_channels=FILTERS1,
                                                kernel_size=5,
                                                stride=1,
                                                padding=2),
                                                
                                            nn.ReLU(),
                                            
                                            nn.Conv2d(in_channels=FILTERS1,
                                                out_channels=FILTERS2,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1),
                                                
                                            nn.ReLU())
                                            
        self.spatial_conv = nn.Sequential(nn.Conv2d(in_channels=2*FILTERS2+self.d3,
                                                out_channels=self.spatial_act_size,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0))
                                            
        self.nonspatial_layers = nn.Sequential(nn.Linear(self.h*self.w*(2*FILTERS2+self.nonspatial_size), FC1SIZE),
                                            nn.ReLU(),
                                            nn.Linear(FC1SIZE, FC2SIZE),
                                            nn.ReLU())
                                            
        self.nonspatial_action_layer = nn.Sequential(nn.Linear(FC2SIZE, self.nonspatial_act_size),
                                                       nn.Softmax())
                                                       
        self.value_layer = nn.Linear(FC2SIZE, 1)
        
    
    def forward(self, screen, minimap, nonspatial_in):
        '''
            screen: (n,17,84,84)
            minimap: (n,7,84,84)
            nonspatial_in: (n,1,1,11)
        '''
        n = nonspatial_in.shape[0]
        
        screen_indices, screen_vals = screen
        screen_indices = torch.from_numpy(screen_indices).long().to(self.device)
        screen_vals = torch.from_numpy(screen_vals).float().to(self.device)
        screen = torch.cuda.sparse.FloatTensor(screen_indices, screen_vals, torch.Size([n, self.d1, self.h, self.w])).to_dense()
        
        minimap_indices, minimap_vals = minimap
        minimap_indices = torch.from_numpy(minimap_indices).long().to(self.device)
        minimap_vals = torch.from_numpy(minimap_vals).float().to(self.device)
        minimap = torch.cuda.sparse.FloatTensor(minimap_indices, minimap_vals, torch.Size([n, self.d2, self.h, self.w])).to_dense()
        
        nonspatial_in = torch.from_numpy(nonspatial_in).float().to(self.device)

        features = self.forward_features(screen, minimap, nonspatial_in)
        
        spatial_policy = self.forward_spatial(features)
        nonspatial_policy, value = self.forward_nonspatial_value(features)
        
        return spatial_policy, nonspatial_policy, value
        
        
    def forward_features(self, screen, minimap, nonspatial_in):
    
        n = screen.shape[0]
        nonspatial_tospatial = torch.ones(n,1,self.h,self.w).float().to(self.device) * nonspatial_in
    
        embedded_screen = self.screen_embed(screen)
        screen_out = self.screen_convs(embedded_screen)
        
        embedded_minimap = self.minimap_embed(minimap)
        minimap_out = self.minimap_convs(embedded_minimap)
        concat_output = torch.cat([screen_out, minimap_out, nonspatial_tospatial], dim=1)
        
        return concat_output
        
        
    def forward_spatial(self, features):
        x = self.spatial_conv(features)
        
        flattened = x.reshape((x.size(0), x.size(1), self.h*self.w))
        flattened = F.softmax(flattened, 2)
        
        x = flattened.reshape(x.shape)
        
        return x
        
    def forward_nonspatial_value(self, features):
        x = features.view(features.size(0), -1)
        x = self.nonspatial_layers(x)
        nonspatial = self.nonspatial_action_layer(x)
        value = self.value_layer(x)
        return nonspatial, value
        
    
    def choose(self, spatial_probs, nonspatial_probs):
        '''
            Chooses a random action for spatial1, spatial2, and nonspatial based on probs.
            
            params:
                spatial_probs: (1,h,w,self.spatial_act_size)
                nonspatial_probs: (1,self.nonspatial_act_size)
        '''
        spatials = []
        for i in range(spatial_probs.shape[1]):
            probs = spatial_probs[0,i,:,:]
            [y,x] = choice = self.choose_action(probs)
            spatials.append([x,y])
            
       
        probs = nonspatial_probs.flatten().cpu().data.numpy()
        nonspatial = self.choose_action(probs)
        return spatials, nonspatial
        
    def choose_action(self, probs):
        choice = random.random()
        if (len(probs.shape) == 2):
            
            probs = self.cumsum2D(probs, choice)            
            probmap = torch.where(probs <= choice, self.where_yes, self.where_no)
            return probmap.nonzero()[-1]
            
        cumsum = np.cumsum(probs)
        output = bisect.bisect(cumsum, choice)
        return output
        
    def cumsum2D(self, probs, choice):
        rowsums = torch.cumsum(torch.sum(probs, 1), 0).reshape((self.h,1))[:-1]
        cumsums = torch.cumsum(probs, 1)
        cumsums[1:, :] += rowsums
        probs[-1,-1] = 1.0
        return cumsums
        
        
        
        
            
        
        
        



















