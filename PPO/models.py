
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from config import *
import numpy as np
import time
import bisect
import random


class GraphConvNet(nn.Module):

    def __init__(self, nonspatial_act_size, spatial_act_size, device):
        super(GraphConvNet, self).__init__()
        self.device = device
        self.nonspatial_act_size = nonspatial_act_size
        self.spatial_act_size = spatial_act_size
        
        self.config = GraphConvConfigMinigames
        self.spatial_width = self.config.spatial_width
        self.fc1_size = self.fc2_size = 100
        self.fc3_size = 50
        
        FILTERS1 = 16
        FILTERS2 = 32
        FILTERS3 = 32
        
        self.W1 = nn.Linear(self.config.unit_vec_width, self.fc1_size)
        self.W2 = nn.Linear(self.fc1_size, self.fc2_size)
        self.W3 = nn.Linear(self.fc2_size, self.fc3_size)
        
        self.action_choice = nn.Linear(self.fc3_size, nonspatial_act_size)
        
        self.value_layer = nn.Linear(self.fc3_size, 1)
        
        
        
        self.tconv1 = torch.nn.ConvTranspose2d(self.fc3_size, 
                                                    FILTERS1,
                                                    kernel_size=4,
                                                    padding=0,
                                                    stride=2)
                                                    
        self.tconv2 = torch.nn.ConvTranspose2d(FILTERS1,
                                                    FILTERS2,
                                                    kernel_size=4,
                                                    padding=1,
                                                    stride=2)
        
        self.tconv3 = torch.nn.ConvTranspose2d(FILTERS2,
                                                    FILTERS3,
                                                    kernel_size=4,
                                                    padding=1,
                                                    stride=2)
                                                    
        self.tconv4 = torch.nn.ConvTranspose2d(FILTERS3,
                                                    self.spatial_act_size,
                                                    kernel_size=4,
                                                    padding=1,
                                                    stride=2)
        
        
                                                    
        self.activation = nn.ReLU()
        
        
    """
        G: (N, graph_n, graph_n)
        X: (N, graph_n, unit_vec_width)
        avail_actions: (N, action_space)
    """
    def forward(self, G, X, avail_actions, choosing=False):
    
        (N, graph_n, _) = G.shape
        
        G = torch.from_numpy(G).to(self.device).float()
        X = torch.from_numpy(X).to(self.device).float()
        avail_actions = torch.from_numpy(avail_actions).float().to(self.device)
        
        A = G + torch.eye(self.config.graph_n).to(self.device).unsqueeze(0)
        D = torch.zeros(A.shape).to(self.device)
        D[:,range(self.config.graph_n), range(self.config.graph_n)] = torch.sum(A, 2)
        
        D_inv_sqrt = D ** -0.5
        A_agg = torch.matmul(torch.matmul(D_inv_sqrt, A), D_inv_sqrt)
        
        h1 = self.activation(torch.matmul(A_agg, self.W1(X)))
        h2 = self.activation(torch.matmul(A_agg, self.W2(h1)))
        h3 = self.activation(torch.matmul(A_agg, self.W3(h2)))
        
        nonspatial_policy = F.softmax(avail_actions * self.action_choice(h3))
        
        value_in = torch.mean(h3, dim=1)
        value = self.value_layer(h3)
        
        stacked_h3 = h3.reshape((N*graph_n, self.fc3_size, 1, 1))
        
        s1 = self.activation(self.tconv1(stacked_h3))
        s2 = self.activation(self.tconv2(s1))
        s3 = self.activation(self.tconv3(s2))
        spatial_policy = F.softmax(self.tconv4(s3).reshape((N*graph_n, self.spatial_act_size, -1)), dim=2).reshape((N, graph_n, self.spatial_act_size, self.spatial_width, self.spatial_width))
        
        choice = None
        if (choosing):
            assert(N==1)
            #choices = self.choose(spatial_policy, nonspatial_policy)
            nonspatial_choice = self.multi_agent_choose_action(nonspatial_policy.detach().cpu().reshape((graph_n, self.nonspatial_act_size)).numpy())
            
            spatial_choice = self.multi_agent_choose_action(spatial_policy.detach().cpu().reshape((graph_n * self.spatial_act_size, self.spatial_width ** 2)).numpy())
            spatial_choice = spatial_choice.reshape((graph_n, self.spatial_act_size))
            
            spatial_out = np.zeros((graph_n, 2*self.spatial_act_size))
            for i in range(self.spatial_act_size):
                spatial_out[:,2*i] = (spatial_choice[:,i] / self.spatial_width).astype(np.int)
                spatial_out[:,2*i+1] = spatial_choice[:,i] % self.spatial_width
            
            choice = [spatial_out, nonspatial_choice]
        
        return spatial_policy, nonspatial_policy, value, choice
        
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
            try:
                coords = probmap.nonzero()[-1]
            except: 
                coords = [0,0]
            return coords
            
        cumsum = np.cumsum(probs)
        output = bisect.bisect(cumsum, choice)
        return output
        
    def cumsum2D(self, probs, choice):
        rowsums = torch.cumsum(torch.sum(probs, 1), 0).reshape((self.h,1))[:-1]
        cumsums = torch.cumsum(probs, 1)
        cumsums[1:, :] += rowsums
        probs[-1,-1] = 1.0
        return cumsums
        
    """
        probs: (graph_n, self.spatial_act_size)
    """
    def multi_agent_choose_action(self, probs):
        (prob_n, _) = probs.shape
        cums = np.cumsum(probs, 1)
        vals = np.random.random(prob_n)
        choices = []
        for i in range(prob_n):
            row = probs[i]
            choices.append(bisect.bisect(row, vals[i]))
        return np.array(choices)
            
    
            























































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
                                            
        self.nonspatial_action_layer = nn.Linear(FC2SIZE, self.nonspatial_act_size)
                                                       
        self.value_layer = nn.Linear(FC2SIZE, 1)
        
    
    def forward(self, screen, minimap, nonspatial_in, avail_actions, history=[], choosing=False):
        '''
            screen: (n,1968,84,84) in coo format
            minimap: (n,?,84,84) in coo format
            nonspatial_in: (n,11, 1, 1)
            avail_actions: (n,5)
            history: list of hist_size previous frames (ignore until LSTM implemented)
        '''
        n = nonspatial_in.shape[0]
        
        screen_indices, screen_vals = screen
        #print(screen_indices.shape)
        screen_indices = torch.from_numpy(screen_indices).long().to(self.device)
        screen_vals = torch.from_numpy(screen_vals).float().to(self.device)
        screen = torch.cuda.sparse.FloatTensor(screen_indices, screen_vals, torch.Size([n, self.d1, self.h, self.w])).to_dense()
        
        minimap_indices, minimap_vals = minimap
        minimap_indices = torch.from_numpy(minimap_indices).long().to(self.device)
        minimap_vals = torch.from_numpy(minimap_vals).float().to(self.device)
        minimap = torch.cuda.sparse.FloatTensor(minimap_indices, minimap_vals, torch.Size([n, self.d2, self.h, self.w])).to_dense()
        
        nonspatial_in = torch.from_numpy(nonspatial_in).float().to(self.device)
        available_actions = torch.from_numpy(avail_actions).byte().to(self.device)
        features = self.forward_features(screen, minimap, nonspatial_in)
        
        spatial_policy = self.forward_spatial(features)
        nonspatial_policy, value = self.forward_nonspatial_value(features, available_actions)
        choice = None
        if (choosing):
            choices = self.choose(spatial_policy, nonspatial_policy)
        
        
        return spatial_policy, nonspatial_policy, value, choices
        
        
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
        
    def forward_nonspatial_value(self, features, avail_actions):
        x = features.view(features.size(0), -1)
        x = self.nonspatial_layers(x)
        
        nonspatial = self.nonspatial_action_layer(x)
        nonspatial.masked_fill_(1-avail_actions, float('-inf'))
        nonspatial = F.softmax(nonspatial, dim=1)
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
            try:
                coords = probmap.nonzero()[-1]
            except: 
                coords = [0,0]
            return coords
            
        cumsum = np.cumsum(probs)
        output = bisect.bisect(cumsum, choice)
        return output
        
    def cumsum2D(self, probs, choice):
        rowsums = torch.cumsum(torch.sum(probs, 1), 0).reshape((self.h,1))[:-1]
        cumsums = torch.cumsum(probs, 1)
        cumsums[1:, :] += rowsums
        probs[-1,-1] = 1.0
        return cumsums
        
        
        
        
            
        
        
        



















