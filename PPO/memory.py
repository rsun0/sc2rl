from config import *
from collections import deque
import utils
import numpy as np
import random
import copy

class ReplayMemory(object):
    def __init__(self, mem_cap, hist_size, batch_size):
        self.memory = deque(maxlen=mem_cap)
        self.nonspatial_action_space = GraphConvConfigMinigames.action_space
        self.spatial_action_width = GraphConvConfigMinigames.spatial_width
        self.access_num = 0
        self.batch_size = batch_size
        self.reset_num = int(mem_cap / batch_size)
        self.indices = []
        self.Memory_capacity = mem_cap
        self.history_size = hist_size
        self.update_indices()
    
    def push(self, history, action, reward, done, vtarg, ret, adv, step):
        # history, action, reward, done, vtarg, adv
        self.memory.append([history, action, reward, done, vtarg, ret, adv, step])
        
        
    def update_indices(self):
        self.indices = list(range(1, self.Memory_capacity - (self.history_size-1)))
        random.shuffle(self.indices)

    def sample_mini_batch(self, frame, hist_size):
        
        
        mini_batch = []
        if frame >= self.Memory_capacity:
            sample_range = self.Memory_capacity
        else:
            sample_range = frame

        
            

        # history size
        
        lower = self.batch_size*self.access_num
        upper = min((self.batch_size*(self.access_num+1)), sample_range)

        idx_sample = self.indices[lower:upper]
        states = []
        for i in idx_sample:
            sample = []
            G_samp = []
            X_samp = []
            avail_samp = []
            hidden_samp = []
            prev_action_samp = []
            
            # Boolean array, corresponds to 1 if frame is valid, 0 if frame is from previous episode that ends between i and i+self.history_size
            relevant_frame = []
            
            
            for j in range(self.history_size):
                sample.append(self.memory[i + j])
                #print("\n", self.memory[i+j], "\n")
                if (self.memory[i+j][-1] == 1):
                    for k in range(j):
                    
                        G_samp[k] = np.zeros(G_samp[k].shape)
                        X_samp[k] = np.zeros(X_samp[k].shape)
                        avail_samp[k] = np.zeros(avail_samp[k].shape)
                        hidden_samp[k] = np.zeros(hidden_samp[k].shape)
                        action_arr[k] = np.zeros(action_arr[k].shape)
                        relevant_frame[k] = 0
                        
                    
                G_samp.append(self.memory[i+j][0][0][0])
                X_samp.append(self.memory[i+j][0][1][0])
                avail_samp.append(self.memory[i+j][0][2])
                hidden_samp.append(self.memory[i+j][0][3])
                
                #print("avail_actions here: ", self.memory[i+j][0][2].shape)
                
                action_arr = utils.action_to_onehot(self.memory[i+j][1], self.nonspatial_action_space, self.spatial_action_width)[0]
                        
                prev_action_samp.append(action_arr)
                
                relevant_frame.append(1)
                
                
            G_samp = np.array(G_samp)
            X_samp = np.array(X_samp)
            avail_samp = np.array(avail_samp)
            hidden_samp = np.array(hidden_samp)
            prev_action_samp = np.array(prev_action_samp)
            relevant_frame = np.array(relevant_frame)
            
            #sample = np.array(sample)
            row = copy.deepcopy(sample[self.history_size-1])
            #print(row)
            #print(sample.shape, row.shape, sample[:,0].shape, sample[0,:].shape, sample[:,0][0].shape, sample[0].shape, type(sample[:,0]), type(sample[:,0][0]))
            #print(avail_samp.shape)
            row[0] = np.array([G_samp, X_samp, avail_samp[-1], hidden_samp[0], prev_action_samp, relevant_frame])
            mini_batch.append(row)


        self.access_num = (self.access_num + 1) % self.reset_num
        if (self.access_num == 0):
            self.update_indices()

        return mini_batch
        
    def compute_vtargets_adv(self, gamma, lam, frame_next_val):
        N = len(self)
        
        prev_gae_t = 0
       
        
        for i in reversed(range(N)):
            
            if i+1 == N:
                vnext = frame_next_val
                nonterminal = 1
            else:
                vnext = self.memory[i+1][4]
                nonterminal = 1 - self.memory[i+1][3]    # 1 - done
            delta = self.memory[i][2] + gamma * vnext * nonterminal - self.memory[i][4]
            gae_t = delta + gamma * lam * nonterminal * prev_gae_t
            self.memory[i][6] = gae_t    # advantage
            self.memory[i][5] = gae_t + self.memory[i][4]  # advantage + value
            prev_gae_t = gae_t
        

    def __len__(self):
        return len(self.memory)
