# OBJECTIVE: maximize theta in SUM (n 1->N) ( pi_theta(an | sn) / pi_theta_old(an | sn) * ADVn ) - C * KL(pi_theta, pi_theta_old)

# for i = 1,2, ...
#   run pi_theta for T timesteps
#   estimate adv function for all timesteps using NN
#   do SGD on objective
#   (consequence: if KL too high, increasing B. if KL too low, decrease B)

# based on code from https://github.com/wooridle/DeepRL-PPO-tutorial/blob/master/ppo.py


import numpy as np
from custom_env import MinigameEnvironment
from modified_state_space import state_modifier
import random
from time import sleep
import matplotlib
import matplotlib.pyplot as plt
import action_interface
import copy
from collections import deque
import custom_env
import models

import torch
import torch.nn as nn
import torch.optim as optim
from memory import ReplayMemory


np.set_printoptions(linewidth=200, precision=4)


class PPOAgent(object):
    def __init__(self, env, lr, hist_size=2, train_step=256, trainable=True):
        
        self.filters1 = 16
        self.filters2 = 32
        self.filters3 = 64
        self.lr = lr
        self.hist_size = hist_size
        self.train_step = train_step
        self.clip_param = 0.2
        self.episodes = 10000
        self.save_frame = 50000
        self.evaluation_reward_length = 100
        self.epochs = 3
        self.num_epochs_trained = 0
        self.discount_factor = 0.99
        self.lam = 0.95
        self.batch_size = 32
        
        
        
        self.env = env
        nonspatial_act_size, spatial_act_depth = env.action_space
        self.nonspatial_act_size, self.spatial_act_depth = env.action_space
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.net = models.DeepMind2017Net(nonspatial_act_size, spatial_act_depth, self.device).to(self.device)
        self.target_net = models.DeepMind2017Net(nonspatial_act_size, spatial_act_depth, self.device).to(self.device)
        
        self.memory = ReplayMemory(self.train_step, self.hist_size, self.batch_size)
        self.optimizer = optim.Adam(params = self.net.parameters(), lr=self.lr)
        
    def update_target_net(self):
        self.target_net.load_state_dict(self.net.state_dict())
        
    def train(self):
    
        evaluation_reward = deque(maxlen=self.evaluation_reward_length)
    
        ### Keep track of average episode rewards, episode values
        rewards, episodes = [], []
    
        ### Keeps track of number of frames seen by agent training        
        frame = 0
        
        for e in range(self.episodes):

            done = False
            score = 0

            ### Stores previous outputs of convolutional map (useless until LSTM implemented)            
            history = []
            
            ### Keeps track of length of current game
            step = 0
            
            state, reward, done, _ = self.env.reset()
            info = state[3]
            screen, minimap, nonspatial_in, avail_actions = state
            
            while not done:
                # Handle selection, edge cases
                if not info['friendly_units_present']:
                    state, reward, done, info = self.env.step(4)
                    continue
                if self._select_next or not info['units_selected']:
                    self._select_next = False
                    state, reward, done, info = self.env.step(0)
                    continue
            
                step += 1
                frame += 1
                
                ### stack history
                recent_hist = self.get_recent_hist(history)
                
                ### Select action, value
                _, _, value, action = self.net(screen, minimap, nonspatial_in, avail_actions, history=recent_hist, choosing=True)
                value = value.cpu().data.numpy().item()
                
                spatial_action, nonspatial_action = action
                
                ### Env step
                state, reward, done, info = self.env.step(nonspatial_action, spatial_action[0], spatial_action[1])
                score += reward
                ### Append state to history
                history.append(state)
                
                ### Store transition in memory
                self.memory.push(state, action, reward, done, value, 0, 0)
                
                ### Start training after random sample generation
                if (frame % self.train_step == 0):
                    for i in range(self.epochs):
                        _, _, frame_next_val, _ = self.net(screen, minimap, nonspatial_in, avail_actions, history=self.get_recent_hist(history), choosing=True)
                        frame_next_val = frame_next_val.cpu().data.numpy().item()
                        self.train_policy_net_ppo(frame, frame_next_val)
                        
                        self.update_target_net()
                
                ### Save model, print time, record information
                if (frame % self.save_frame == 0):
                    print('now time : ', datetime.now())
                    rewards.append(np.mean(evaluation_reward))
                    episodes.append(e)
                    plt.plot(episodes, rewards, 'r')
                    plt.save_fig("save_model/Starcraft2" + self.env.map + "PPOgraph.png")
                    torch.save(self.net.state_dict(), "save_model/Starcraft2" + self.env.map + "PPO")
                
                ### Handle end of game logic    
                if done:
                    evaluation_reward.append(score)
                    print("episode:", e, "  score:", score, "  steps:", step, "  evaluation reward:", np.mean(evaluation_reward))
                    state, reward, done, _ = self.env.reset()
                    
                screen, minimap, nonspatial_in, avail_actions = state
                    
    ### Main training logic            
    def train_policy_net_ppo(self, frame, frame_next_val):
        
        for param_group in self.optimizer.param_groups:
            curr_lr = param_group['lr']
        print("Training network. lr: %f. clip: %f" % (curr_lr, self.clip_param))
        
        ### Compute value targets and advantage for all frames
        self.memory.compute_vtargets_adv(self.discount_factor, self.lam, frame_next_val)

        ### number of iterations of batches of size self.batch_size. Should divide evenly
        num_iters = int(len(self.memory) / self.batch_size)

        ### Do multiple epochs
        for i in range(self.epochs):
            
            pol_loss = 0.0
            vf_loss = 0.0
            ent_total = 0.0
            
            self.num_epochs_trained += 1
            
            for i in range(num_iters):
                
                mini_batch = self.memory.sample_mini_batch(frame)
                mini_batch = np.array(mini_batch).transpose()
                
                ### Shape: (batch_size, hist_size, 2)
                states = np.stack(mini_batch[0], axis=0)
                
                
                
                
    
    
    def get_recent_hist(self, hist):
        length = min(len(hist), self.hist_size)
        if (length == 0):
            return []
        else:
            return hist[-length:]
                        
        

def main():
    env = custom_env.MinigameEnvironment(state_modifier.modified_state_space,
                                            map_name_="DefeatRoaches",
                                            render=False,
                                            step_multiplier=8)
    lr = 0.00025                       
    agent = PPOAgent(env, lr)
    agent.train()


if __name__ == "__main__":
    main()












