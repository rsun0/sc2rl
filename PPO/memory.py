from config import *
from collections import deque
import numpy as np
import random


class ReplayMemory(object):
    def __init__(self, mem_cap, hist_size, batch_size):
        self.memory = deque(maxlen=mem_cap)
        self.access_num = 0
        self.batch_size = batch_size
        self.reset_num = int(mem_cap / batch_size)
        self.indices = []
        self.Memory_capacity = mem_cap
        self.history_size = hist_size
        self.update_indices()
    
    def push(self, history, action, reward, done, vtarg, ret, adv):
        # history, action, reward, done, vtarg, adv
        self.memory.append([history, action, reward, done, vtarg, ret, adv])
        
        
    def update_indices(self):
        self.indices = list(range(self.Memory_capacity - (self.history_size)))
        random.shuffle(self.indices)

    def sample_mini_batch(self, frame):
        
        
        
        
        mini_batch = []
        if frame >= self.Memory_capacity:
            sample_range = self.Memory_capacity
        else:
            sample_range = frame

        
            

        # history size
        sample_range -= (self.history_size-1)
        
        lower = self.batch_size*self.access_num
        upper = min((self.batch_size*(self.access_num+1)), sample_range)

        idx_sample = self.indices[lower:upper]
        for i in idx_sample:
            sample = []
            for j in range(self.history_size):
                sample.append(self.memory[i + j])

            #sample = np.array(sample)
            row = sample[self.history_size-1]
            for i in range(len(sample)):
                print(len(sample), len(sample[i]), len(sample[i][0]))
                print(sample[i][0].shape)
            #print(sample.shape, row.shape, sample[:,0].shape, sample[0,:].shape, sample[:,0][0].shape, sample[0].shape, type(sample[:,0]), type(sample[:,0][0]))
            row[0] = np.array(sample[:,0])
            mini_batch.append(np.array(row))

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
