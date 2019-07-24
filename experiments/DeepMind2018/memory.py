from collections import deque
import numpy as np
import random
import copy
from sc2env_utils import env_config

class ReplayMemory(object):
    def __init__(self, mem_cap, batch_size, hist_size=1):
        self.memory = deque(maxlen=mem_cap)
        self.nonspatial_action_space = 84
        self.spatial_action_width = 84
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
        self.indices = list(range(1, self.Memory_capacity - (self.history_size)))
        random.shuffle(self.indices)

    def sample_mini_batch(self, frame, hist_size=1):

        if frame >= self.Memory_capacity:
            sample_range = self.Memory_capacity-1
        else:
            sample_range = frame




        # history size

        lower = self.batch_size*self.access_num
        upper = min((self.batch_size*(self.access_num+1)), sample_range)

        idx_sample = self.indices[lower:upper]
        mini_batch = []
        for i in idx_sample:

            if (i == 0):
                prev_action = 0
            else:
                prev_action = np.array([self.memory[i-1][1][0]])
            row = np.array(copy.deepcopy(self.memory[i]))
            (minimap, screen, player, avail, hidden) = row[0]
            row[0] = np.array([np.array(minimap), np.array(screen), np.array(player), np.array(avail), np.array(hidden), np.array(prev_action)])
            row[1][0] = np.array([row[1][0]])
            row[1] = np.array(row[1])
            mini_batch.append(row)

        self.access_num = (self.access_num + 1) % self.reset_num
        if (self.access_num == 0):
            self.update_indices()

        return mini_batch

    def compute_vtargets_adv(self, gamma, lam):
        N = len(self)

        prev_gae_t = 0


        for i in reversed(range(N-1)):

            vnext = self.memory[i+1][4]
            nonterminal = 1 - self.memory[i+1][3]    # 1 - done
            delta = self.memory[i][2] + gamma * vnext * nonterminal - self.memory[i][4]
            gae_t = delta + gamma * lam * nonterminal * prev_gae_t
            self.memory[i][6] = gae_t    # advantage
            self.memory[i][5] = gae_t + self.memory[i][4]  # advantage + value
            prev_gae_t = gae_t


    def __len__(self):
        return len(self.memory)
