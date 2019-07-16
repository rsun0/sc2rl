
import sys
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from config import *
import numpy as np
import time
import bisect
import random
from agent import Model

class GraphConvModel(nn.Module, Model):
    def __init__(self, nonspatial_act_size, spatial_act_size, device="cpu"):
        # Calls nn.Module's constructor, probably
        super().__init__()
        self.nonspatial_act_size = nonspatial_act_size
        self.spatial_act_size = spatial_act_size
        self.device = device

        self.config = GraphConvConfigMinigames 
        self.spatial_width = self.config.spatial_width
        self.embed_size = 256
        self.fc1_size = 256
        self.fc2_size = 256
        self.fc3_size = 256
        self.action_size = nonspatial_act_size + 4
        self.hidden_size = 256
        self.action_fcsize = 256
        self.graph_lstm_out_size = self.hidden_size + self.fc3_size + self.action_size
        self.curr_LSTM_val = None
        
        FILTERS1 = 256
        FILTERS2 = 128
        FILTERS3 = 64
        FILTERS4 = 64
        FILTERS5 = 64
        
        self.where_yes = torch.ones(1).to(self.device)
        self.where_no = torch.zeros(1).to(self.device)
        
        self.unit_embedding = nn.Linear(self.config.unit_vec_width, self.embed_size)
        self.W1 = nn.Linear(self.embed_size, self.fc1_size)
        self.W2 = nn.Linear(self.fc1_size, self.fc2_size)
        self.W3 = nn.Linear(self.fc2_size, self.fc3_size)
        
        #self.fc1 = nn.Linear(self.hidden_size, self.action_fcsize)
        self.action_choice = nn.Linear(self.graph_lstm_out_size, nonspatial_act_size)
        
        self.value_layer = nn.Linear(self.graph_lstm_out_size, 1)
        
        
        
        
        
        self.tconv1 = torch.nn.ConvTranspose2d(self.graph_lstm_out_size, 
                                                    FILTERS1,
                                                    kernel_size=4,
                                                    padding=0,
                                                    stride=2)
                                                    
        
        self.conv2 = torch.nn.Conv2d(FILTERS1,
                                        FILTERS2,
                                        kernel_size=3,
                                        padding=1,
                                        stride=1)
        
        self.conv3 = torch.nn.Conv2d(FILTERS2,
                                        FILTERS3,
                                        kernel_size=3,
                                        padding=1,
                                        stride=1)
        
        
        self.conv4 = torch.nn.Conv2d(FILTERS3,
                                        FILTERS4,
                                        kernel_size=3,
                                        padding=1,
                                        stride=1)
                                        
        self.conv5 = torch.nn.Conv2d(FILTERS4,
                                        FILTERS5,
                                        kernel_size=3,
                                        padding=1,
                                        stride=1)
                                        
        self.conv6 = torch.nn.Conv2d(FILTERS5,
                                        self.spatial_act_size,
                                        kernel_size=3,
                                        padding=1,
                                        stride=1)

        self.activation = nn.Tanh()
        self.conv_activation = nn.ReLU()
        
        self.LSTM_embed_in = nn.Linear(self.fc3_size+self.action_size, self.hidden_size)
        
        self.hidden_layer = nn.LSTM(input_size=self.hidden_size,
                                        hidden_size=self.hidden_size,
                                        num_layers=1)
                                        

    def forward(self, G, X, avail_actions, LSTM_hidden, prev_actions, relevant_frames=np.array([[1]]), epsilon=0.0, choosing=False, selecting=np.array([[1]])):
    
        self.curr_LSTM_val = LSTM_hidden
    
        rand = random.random()
    
        (N, _, graph_n, _) = G.shape
        
        G = torch.from_numpy(G).to(self.device).float()
        X = torch.from_numpy(X).to(self.device).float()
        avail_actions = torch.from_numpy(avail_actions).byte().to(self.device)
        LSTM_hidden = torch.from_numpy(LSTM_hidden).to(self.device).float()
        prev_actions = torch.from_numpy(prev_actions).to(self.device).float()
        relevant_frames = torch.from_numpy(relevant_frames).to(self.device).float()
        selecting = torch.from_numpy(selecting).to(self.device).float()
        
        
        # Shape is (2, graph_n, N, hidden_size) after this
        LSTM_hidden = LSTM_hidden.expand(-1, graph_n, N, -1)
        # Shape is (2, 1, graph_n*N, hidden_size) after this
        LSTM_hidden = LSTM_hidden.flatten(start_dim=1,end_dim=2).unsqueeze(1)
        
        LSTM_graph_out, LSTM_hidden = self.graph_LSTM_forward(G, X, LSTM_hidden, prev_actions, relevant_frames, selecting)
        
        spatial_policy, nonspatial_policy, attack_value, attack_choice = self.attack(LSTM_graph_out, avail_actions, selecting)
        
        select_policy, select_value, select_choice = self.select(LSTM_graph_out,
                                                                    selecting)
                                                                    
        value, LSTM_hidden, choice = self.mesh_select_attack(attack_value, select_value,
                                                            LSTM_hidden,
                                                            attack_choice, select_choice,
                                                            selecting)
        
        ### @TODO: Implement self.attack, self.select, self.mesh_select_attack
        
        return spatial_policy, nonspatial_policy, value, LSTM_hidden, choice
        
        
            

    def graph_LSTM_forward(self, G, X, LSTM_hidden, prev_actions, relevant_frames):
        
        batch_size, D = G.shape[0], G.shape[1]
        graph_out_actions = None
        for i in range(D):
            G_curr = G[:,i,:,:]
            X_curr = X[:,i,:,:]
            relevance = relevant_frames[:,i]

            # shape is (N, graph_n, fc3_size)            
            graph_out = self.graph_forward(G_curr, X_curr)
            expanded_prev_actions = prev_actions[:,[i],:].expand(batch_size,
                                                                    self.config.graph_n,
                                                                    -1)
            
            # select_moves, attack_moves are (N,) shape one-hot vecs
            # select_moves[j] is 1 iff prev_actions[j,i] corresponds to selection
            # attack_moves[j] is 1 iff prev_actions[j,i] corresponds to attack
            select_moves, attack_moves = self.parse_action_phases(expanded_prev_actions)

            graph_out_actions = torch.cat([graph_out, expanded_prev_actions], dim=-1)
               
            # (N, graph_n, -1)                                                     
            embedded_graph = self.activation(self.LSTM_embed_in(graph_out_actions))
            
            """
            # shape is (1, N*graph_n, embed_size) after next line
            embedded_graph = embedded_graph.unsqueeze(0).flatten(start_dim=1, end_dim=2)
            """
            
            # @TODO: Implement attack_LSTM, select_LSTM, merge_select_attack
            
            attack_output, attack_LSTM_hidden = self.attack_LSTM(embedded_graph,
                                                                LSTM_hidden,
                                                                expanded_prev_actions,   
                                                                attack_moves)
            
            select_output, select_LSTM_hidden = self.select_LSTM(embedded_graph,
                                                                LSTM_hidden,
                                                                expanded_prev_actions,
                                                                select_moves)
                                                               
            # output: (1, N, graph_n, -1). LSTM_hidden: (2, 1, N, graph_n, -1) 
            output, LSTM_hidden = self.merge_select_attack(attack_output,
                                                            attack_LSTM_hidden,
                                                            attack_moves,
                                                            select_output,
                                                            select_LSTM_hidden,
                                                            select_moves
                                                            )
            
            """
            # embedded graph: (1, N*graph_n, embed_size)
            # LSTM_hidden: (2, 1, N*graph_n, hidden_size)
            output, LSTM_hidden = self.hidden_layer(embedded_graph, tuple(LSTM_hidden))
            LSTM_hidden = torch.stack(LSTM_hidden)
            """
            
            irrelevant_mask = relevance == 0
            if (irrelevant_mask.size() != torch.Size([0]) and torch.sum(irrelevant_mask) > 0):
                LSTM_hidden[:,:,irrelevant_mask,:,:] = self.init_hidden(torch.sum(irrelevant_mask), device=self.device)
                
        output = torch.cat([output.squeeze(0), graph_out_actions], dim=-1)
        return output, LSTM_hidden

    """
        Returns graph output of shape (N, graph_n, fc3_size)
    """
    def graph_forward(self, G, X):
    
        A = G + torch.eye(self.config.graph_n).to(self.device).unsqueeze(0)
        D = torch.zeros(A.shape).to(self.device)
        D[:,range(self.config.graph_n), range(self.config.graph_n)] = torch.max(torch.sum(A, 2), self.where_yes)
        
        D_inv_sqrt = D
        D_inv_sqrt[:,range(self.config.graph_n), range(self.config.graph_n)] = 1 / (D[:,range(self.config.graph_n), range(self.config.graph_n)] ** 0.5)
        
        A_agg = torch.matmul(torch.matmul(D_inv_sqrt, A), D_inv_sqrt)
        
        embedding = self.activation(self.unit_embedding(X))
        h1 = self.activation(torch.matmul(A_agg, self.W1(embedding)))
        h2 = self.activation(torch.matmul(A_agg, self.W2(h1)))
        h3 = self.activation(torch.matmul(A_agg, self.W3(h2)))
        
        graph_conv_out = h3
        
        return graph_conv_out
        
        
    def attack_LSTM(self,
                    embedded_graph,
                    LSTM_hidden,
                    expanded_prev_actions,   
                    attack_moves):
            
    def select_LSTM(self,
                    embedded_graph,
                    LSTM_hidden,
                    expanded_prev_actions,
                    select_moves):
        
                    
        
    # @TODO: Implement
    def merge_select_attack(self, 
                            attack_output,
                            attack_LSTM_hidden,
                            attack_moves,
                            select_output,
                            select_LSTM_hidden,
                            select_moves):
        raise NotImplementedError
        
    def parse_action_phases(prev_actions):
        

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
                coords = probmap.nonzero()[-1].cpu().data.numpy()
            except: 
                coords = [0,0]
            return coords
            
        cumsum = np.cumsum(probs)
        output = bisect.bisect(cumsum, choice)
        return output
        
    def cumsum2D(self, probs, choice):
        rowsums = torch.cumsum(torch.sum(probs, 1), 0).reshape((self.spatial_width,1))[:-1]
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
        
    def init_hidden(self, batch_size=1, device=None, use_torch=True):
        if (not use_torch):
            return np.zeros((2, 1, batch_size, 1, self.hidden_size))
        return torch.zeros((2, 1, batch_size, 1, self.hidden_size)).float().to(device)
    
    def null_actions(self, batch_size, use_torch=True):
        if (not use_torch):
            return np.zeros((batch_size, self.action_size))
        return torch.zeros((batch_size, self.action_size)).float().to(device)
            



