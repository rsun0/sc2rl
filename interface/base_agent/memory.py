from collections import deque
import numpy as np
import random
import copy
import torch
from base_agent.sc2env_utils import env_config
import matplotlib.pyplot as plt


device = torch.cuda.is_available()

class SequentialMemory(object):
    def __init__(self, mem_cap, batch_size, hist_size=1, hidden_shape=(96,8,8)):
        self.minimaps = torch.zeros((mem_cap,) + env_config["minimap_shape"]).float().to(device)
        self.screens = torch.zeros((mem_cap,) + env_config["screen_shape"]).float().to(device)
        self.players = torch.zeros((mem_cap,) + (env_config["raw_player"],)).float().to(device)
        self.available = torch.zeros((mem_cap,) + (env_config["action_space"],)).float().to(device)
        self.hiddens = torch.zeros((mem_cap,) + hidden_shape).float().to(device)

        self.memory = deque(maxlen=mem_cap)
        self.access_num = 0
        self.batch_size = batch_size
        self.history_size = hist_size
        self.reset_num = int(mem_cap / batch_size)
        self.memory_capacity = mem_cap
        self.indices = []
        self.update_indices()
        self.push_index = 0

    def push(self, state, action, reward, done, vtarg, ret, adv, step):
        minimap, screen, player, avail, hidden = state
        self.minimaps[self.push_index] = torch.from_numpy(minimap).to(device)
        self.screens[self.push_index] = torch.from_numpy(screen).to(device)
        self.players[self.push_index] = torch.from_numpy(player).to(device)
        self.available[self.push_index] = torch.from_numpy(avail).to(device)
        self.hiddens[self.push_index] = torch.from_numpy(hidden).to(device)
        self.memory.append([action, reward, done, vtag, ret, adv, step])
        self.push_index = (self.push_index + 1) % self.memory_capacity

    def update_indices(self):
        last_index = self.memory_capacity - self.batch_size - self.history_size
        self.indices = list(range(0, last_index, self.batch_size)) + [last_index]
        random.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)

    def sample_mini_batch(self, frame, hist_size=self.history_size):

        index = self.indices[frame]
        upper = index + self.batch_size + hist_size
        idx_sample = range(index, upper)
        mini_batch = []

        minimap = self.minimaps[index:upper]
        screen = self.screens[index:upper]
        player = self.players[index:upper]
        avail = self.available[upper-self.batch_size:upper]
        hidden = self.hiddens[index:upper]

        prev_action_sample = []
        relevant_frame = []

        for i in idx_sample:

            row = self.memory[i]
            prev_action = self.memory[i-1][0][0]

            prev_action_sample.append(prev_action)
            if (row[2]):
                relevant_frame.append(1)
            else:
                relevant_frame.append(0)

            row[0] = np.array(row[0])
            #state, action, _, _, _, _, _, _ = row
            #print(state[0].shape, state[1].shape, state[2].shape)
            mini_batch.append(row)

        prev_action_sample = np.array([prev_action_sample])
        relevant_frame = np.array(relevant_frame)

        self.access_num = (self.access_num + 1) % self.reset_num
        if (self.access_num == 0):
            self.update_indices()

        states = np.array([minimap, screen, player, avail, hidden, prev_action_sample, relevant_frame])
        mini_batch = np.array(mini_batch)

        return states, mini_batch

    def compute_vtargets_adv(self, gamma, lam):
        N = len(self)

        prev_gae_t = 0


        for i in reversed(range(N-1)):

            vnext = self.memory[i+1][3]
            nonterminal = 1 - self.memory[i+1][2]    # 1 - done
            delta = self.memory[i][1] + gamma * vnext * nonterminal - self.memory[i][3]
            gae_t = delta + gamma * lam * nonterminal * prev_gae_t
            self.memory[i][5] = gae_t    # advantage
            self.memory[i][4] = gae_t + self.memory[i][3]  # advantage + value
            prev_gae_t = gae_t

    """
        Performs random equivalent reorientation of state
    """
    def random_transform(self, minimap, screen, hidden, action, transform):
        #minimap, screen = state[:, :2]
        #print(state.shape, minimap.shape, screen.shape)
        spatial_args = action
        new_spatial_action = np.copy(spatial_args)
        spatial_w = env_config["spatial_action_size"]

        #transform = np.random.randint(0,8)

        # Rotations
        if transform >= 2 and transform < 4:
            minimap = np.rot90(minimap, k=1, axes=(-2,-1))
            screen = np.rot90(screen, k=1, axes=(-2,-1))
            hidden = np.rot90(hidden, k=1, axes=(-2,-1))
            new_spatial_action[:,0] = (spatial_w - 1) - spatial_args[:,1]
            new_spatial_action[:,1] = spatial_args[:,0]
        if transform >= 4 and transform < 6:
            minimap = np.rot90(minimap, k=2, axes=(-2,-1))
            screen = np.rot90(screen, k=2, axes=(-2,-1))
            hidden = np.rot90(hidden, k=2, axes=(-2,-1))
            new_spatial_action[:,0] = (spatial_w - 1) - spatial_args[:,0]
            new_spatial_action[:,1] = (spatial_w - 1) - spatial_args[:,1]
        elif transform >= 6 and transform < 8:
            minimap = np.rot90(minimap, k=3, axes=(-2,-1))
            screen = np.rot90(screen, k=3, axes=(-2,-1))
            hidden = np.rot90(hidden, k=3, axes=(-2,-1))
            new_spatial_action[:,0] = spatial_args[:,1]
            new_spatial_action[:,1] = (spatial_w - 1) - spatial_args[:,0]

        # Reflection
        if transform % 2 == 1:
            minimap = np.flip(minimap, -1)
            screen = np.flip(screen, -1)
            hidden = np.flip(hidden, -1)
            new_spatial_action[:,1] = (spatial_w - 1) - new_spatial_action[:,1]

        action = new_spatial_action
        return minimap, screen, hidden,  action

    def batch_random_transform(self, minimaps, screens, hiddens, actions):
        for i in range(len(minimaps)):
            transform = np.random.randint(0,8)
            minimaps[i], screens[i], hiddens[i], actions[i] = self.random_transform(minimaps[i], screens[i], hiddens[i], actions[i], transform)
        return minimaps, screens, hiddens, actions




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
        """
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
        """
        for i in idx_sample:
            sample = []
            minimap_sample = []
            screen_sample = []
            player_sample = []
            avail_sample = []
            hidden_sample = []
            prev_action_sample = []
            relevant_frame = []

            for j in range(self.history_size):
                sample.append(self.memory[i+j])

                # if done
                if (self.memory[i+1][-1]):
                    relevant_frame = [0 for k in relevant_frame]

                state = self.memory[i+j][0]
                prev_action = self.memory[i+j-1][1][0]

                minimap_sample.append(state[0])
                screen_sample.append(state[1])
                player_sample.append(state[2])
                avail_sample.append(state[3])
                hidden_sample.append(state[4])
                prev_action_sample.append(prev_action)
                relevant_frame.append(1)

            minimap_sample = np.array(minimap_sample)
            screen_sample = np.array(screen_sample)
            player_sample = np.array(player_sample)
            avail_sample = np.array(avail_sample)
            hidden_sample = np.array(hidden_sample)
            prev_action_sample = np.array([prev_action_sample])
            relevant_frame = np.array(relevant_frame)

            row = copy.deepcopy(sample[self.history_size-1])
            row[0] = np.array([minimap_sample, screen_sample, player_sample, avail_sample[-1], hidden_sample[0], hidden_sample[-1], prev_action_sample, relevant_frame])
            row[1][0] = np.array([row[1][0]])
            row[1] = np.array(row[1])
            #state, action, _, _, _, _, _, _ = row
            #print(state[0].shape, state[1].shape, state[2].shape)
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

    """
        Performs random equivalent reorientation of state
    """
    def random_transform(self, minimap, screen, hidden, action, transform):
        #minimap, screen = state[:, :2]
        #print(state.shape, minimap.shape, screen.shape)
        spatial_args = action
        new_spatial_action = np.copy(spatial_args)
        spatial_w = env_config["spatial_action_size"]

        #transform = np.random.randint(0,8)

        # Rotations
        if transform >= 2 and transform < 4:
            minimap = np.rot90(minimap, k=1, axes=(-2,-1))
            screen = np.rot90(screen, k=1, axes=(-2,-1))
            hidden = np.rot90(hidden, k=1, axes=(-2,-1))
            new_spatial_action[:,0] = (spatial_w - 1) - spatial_args[:,1]
            new_spatial_action[:,1] = spatial_args[:,0]
        if transform >= 4 and transform < 6:
            minimap = np.rot90(minimap, k=2, axes=(-2,-1))
            screen = np.rot90(screen, k=2, axes=(-2,-1))
            hidden = np.rot90(hidden, k=2, axes=(-2,-1))
            new_spatial_action[:,0] = (spatial_w - 1) - spatial_args[:,0]
            new_spatial_action[:,1] = (spatial_w - 1) - spatial_args[:,1]
        elif transform >= 6 and transform < 8:
            minimap = np.rot90(minimap, k=3, axes=(-2,-1))
            screen = np.rot90(screen, k=3, axes=(-2,-1))
            hidden = np.rot90(hidden, k=3, axes=(-2,-1))
            new_spatial_action[:,0] = spatial_args[:,1]
            new_spatial_action[:,1] = (spatial_w - 1) - spatial_args[:,0]

        # Reflection
        if transform % 2 == 1:
            minimap = np.flip(minimap, -1)
            screen = np.flip(screen, -1)
            hidden = np.flip(hidden, -1)
            new_spatial_action[:,1] = (spatial_w - 1) - new_spatial_action[:,1]

        action = new_spatial_action
        return minimap, screen, hidden,  action

    def batch_random_transform(self, minimaps, screens, hiddens, actions):
        for i in range(len(minimaps)):
            transform = np.random.randint(0,8)
            minimaps[i], screens[i], hiddens[i], actions[i] = self.random_transform(minimaps[i], screens[i], hiddens[i], actions[i], transform)
        return minimaps, screens, hiddens, actions


    def __len__(self):
        return len(self.memory)
