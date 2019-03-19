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

from config import *

np.set_printoptions(linewidth=200, precision=4)


class PPOAgent(object):
    def __init__(self, env, lr, hist_size=1, train_step=1024, trainable=True):
        
        self.filters1 = 16
        self.filters2 = 32
        self.filters3 = 64
        self.lr = lr
        self.hist_size = hist_size
        self.train_step = train_step
        self.clip_param = 0.1
        self.eps_denom = 1e-6
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
        self.net = models.GraphConvNet(nonspatial_act_size, spatial_act_depth, self.device).to(self.device)
        self.target_net = models.GraphConvNet(nonspatial_act_size, spatial_act_depth, self.device).to(self.device)
        
        self.memory = ReplayMemory(self.train_step, self.hist_size, self.batch_size)
        self.optimizer = optim.Adam(params = self.net.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        
        self.c1 = 1.0
        self.c2 = 0.01
        
        self.averages = []
        
    def update_target_net(self):
        self.target_net.load_state_dict(self.net.state_dict())
        
    def load_saved_model(self):
        self.net.load_state_dict(torch.load("save_model/Starcraft2" + self.env.map + "PPO"))
        self.update_target_net()
        
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
            score = 0
            
            state, reward, done, info = self.env.reset()
            action = [np.array([[0,0],[0,0]]), 0]
            value = 0
            r = 0
            G, X, avail_actions = state
            _select_next = True
            
            while not done:
                # Handle selection, edge cases
                
                if (not info['friendly_units_present']):
                    print("hello")
                    state, reward, done, info = self.env.step(0)
                    continue
                
                
                step += 1
                frame += 1
                
                ### stack history
                recent_hist = self.get_recent_hist(history)
                
                ### Select action, value
                _, _, value, action = self.net(G, X, avail_actions, choosing=True)
                value = value.cpu().data.numpy().item()
                
                spatial_action, nonspatial_action = action
                                
                #print(action)
                ### Env step

                state, reward, done, info = self.env.step(nonspatial_action, spatial_action[0], spatial_action[1])
                G, X, avail_actions = state
                action = [np.array(spatial_action), nonspatial_action]
                score += reward
                ### Append state to history
                history.append(state)
                    
                ### Store transition in memory
                self.memory.push(state, action, reward, done, value, 0, 0)
                
                ### Start training after random sample generation
                    
                if (frame % self.train_step == 0 and frame != 0):
                    _, _, frame_next_val, _ = self.net(G, X, avail_actions)
                    frame_next_val = frame_next_val.cpu().data.numpy().item()
                    self.train_policy_net_ppo(frame, frame_next_val)
                            
                    self.update_target_net()
                    
                    
                ### Save model, print time, record information
                if (frame % self.save_frame == 0):
                    #print('now time : ', datetime.now())
                    rewards.append(np.mean(evaluation_reward))
                    episodes.append(e)
                    plt.plot(episodes, rewards, 'r')
                    plt.savefig("save_model/Starcraft2" + self.env.map + "PPOgraph.png")
                    torch.save(self.net.state_dict(), "save_model/Starcraft2" + self.env.map + "PPO")
                
                ### Handle end of game logic    
                if done:
                    evaluation_reward.append(score)
                    print("episode:", e, "  score:", score, "  steps:", step, "  evaluation reward:", np.mean(evaluation_reward))
                    #state, reward, done, _ = self.env.reset()
                    self.averages.append(np.mean(evaluation_reward))
                    #self.plot_results()

                    
                G, X, avail_actions = state
                    
    ### Main training logic            
    def train_policy_net_ppo(self, frame, frame_next_val):
        
        for param_group in self.optimizer.param_groups:
            curr_lr = param_group['lr']
        print("\n\n ------- Training network. lr: %f. clip: %f ------- \n\n" % (curr_lr, self.clip_param))
        
        ### Compute value targets and advantage for all frames
        self.memory.compute_vtargets_adv(self.discount_factor, self.lam, frame_next_val)

        ### number of iterations of batches of size self.batch_size. Should divide evenly
        num_iters = int(len(self.memory) / self.batch_size)
        device = self.device
        ### Do multiple epochs
        for i in range(self.epochs):
            
            pol_loss = 0.0
            vf_loss = 0.0
            ent_total = 0.0
            
            self.num_epochs_trained += 1
            
            for j in range(num_iters):
                
                mini_batch = self.memory.sample_mini_batch(frame)
                mini_batch = np.array(mini_batch).transpose()
                
                states = np.stack(mini_batch[0], axis=0)
                G_states = np.stack(states[:,0], axis=1).squeeze(0)
                X_states = np.stack(states[:,1], axis=1).squeeze(0)
                avail_states = np.stack(states[:,2], axis=0)
                
                n = states.shape[0]
                
                actions = np.array(list(mini_batch[1]))
                spatial_actions = np.stack(actions[:,0],0)
                first_spatials = spatial_actions[:,0]
                second_spatials = spatial_actions[:,1]
                nonspatial_acts = np.array(actions[:,1]).astype(np.int64)
                
                
                rewards = np.array(list(mini_batch[2]))
                dones = mini_batch[3]
                v_returns = mini_batch[5].astype(np.float32)
                advantages = mini_batch[6].astype(np.float32)
                
                first_spatials = torch.from_numpy(first_spatials).to(device)
                second_spatials = torch.from_numpy(second_spatials).to(device)
                nonspatial_acts = torch.from_numpy(nonspatial_acts).to(device)
                nonspatial_acts = nonspatial_acts.unsqueeze(1)
                
                rewards = torch.from_numpy(rewards).to(device)
                dones = torch.from_numpy(np.uint8(dones)).to(device)
                v_returns = torch.from_numpy(v_returns).to(device)
                advantages = torch.from_numpy(advantages).to(device)
                
                advantages = (advantages - advantages.mean()) / (torch.clamp(advantages.std(), self.eps_denom))
                
                spatial_probs, nonspatial_probs, values, _ = self.net(G_states, X_states, avail_states)
                old_spatial_probs, old_nonspatial_probs, old_values, _ = self.target_net(G_states, X_states, avail_states)
                
                
                #print(nonspatial_probs.shape, self.index_spatial_probs(spatial_probs[:,0,:,:], first_spatials).shape, (nonspatial_acts < 2).shape)
                #print(nonspatial_probs.shape, nonspatial_acts.shape)
                #print(nonspatial_probs[range(self.batch_size),nonspatial_acts].shape)
                
                gathered_nonspatials = nonspatial_probs.gather(1, nonspatial_acts).squeeze(1)
                old_gathered_nonspatials = old_nonspatial_probs.gather(1, nonspatial_acts).squeeze(1)
                first_spatial_mask = (nonspatial_acts < 3).to(self.device).float().squeeze(1)
                second_spatial_mask = (nonspatial_acts == 0).to(self.device).float().squeeze(1)
                
                numerator = torch.log(gathered_nonspatials + self.eps_denom) + torch.log(self.index_spatial_probs(spatial_probs[:,0,:,:], first_spatials) + self.eps_denom) * first_spatial_mask + (torch.log(self.index_spatial_probs(spatial_probs[:,1,:,:], second_spatials) + self.eps_denom) * second_spatial_mask)
                denom = torch.log(old_gathered_nonspatials + self.eps_denom) + torch.log(self.index_spatial_probs(old_spatial_probs[:,0,:,:], first_spatials) + self.eps_denom) * first_spatial_mask + (torch.log(self.index_spatial_probs(old_spatial_probs[:,1,:,:], second_spatials) + self.eps_denom) * second_spatial_mask)
                
                
                """
                denom = old_gathered_nonspatials
                print(nonspatial_probs.shape)
                print(denom.shape)
                print((nonspatial_acts < 3).shape)
                print(((self.index_spatial_probs(spatial_probs[:,0,:,:], first_spatials)) * (nonspatial_acts < 3).to(self.device).float()).shape)
                denom[nonspatial_acts < 3] = denom[nonspatial_acts < 3] * self.index_spatial_probs(spatial_probs[:,0,:,:], first_spatials)
                denom[nonspatial_acts == 0] = denom[nonspatial_acts == 0] * self.index_spatial_probs(old_spatial_probs[:,1,:,:], second_spatials)
                
                denom = torch.log( torch.clamp( denom, self.eps_denom ) )
                """
                
                
                ratio = torch.exp(numerator - denom)
                ratio_adv = ratio * advantages.detach()
                bounded_adv = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param) * advantages.detach()
                
                """
                print("ratio: ", ratio, "\n\n")
                print("numerator: ", numerator, "\n\n")
                print("denominator: ", denom, "\n\n")
                """
                
                pol_avg = - ((torch.min(ratio_adv, bounded_adv)).mean())
                
                value_loss = self.loss(values.squeeze(1), v_returns.detach())
                
                ent = self.entropy(spatial_probs, nonspatial_probs)
                
                
                total_loss = pol_avg + self.c1 * value_loss - self.c2 * ent
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                pol_loss += pol_avg.detach().item()
                vf_loss += value_loss.detach().item()
                ent_total += ent.detach().item() 
                
            pol_loss /= num_iters
            vf_loss /= num_iters
            ent_total /= num_iters
            print("Iteration %d: Policy loss: %f. Value loss: %f. Entropy: %f" % (self.num_epochs_trained, pol_loss, vf_loss, ent_total))
        
        print("\n\n ------- Training sequence ended ------- \n\n")
                
                              
    def index_spatial_probs(self, spatial_probs, indices):
        index_tuple = torch.meshgrid([torch.arange(x) for x in spatial_probs.size()[:-2]]) + (indices[:,0], indices[:,1],)
        output = spatial_probs[index_tuple]
        return output
                
    def get_recent_hist(self, hist):
        length = min(len(hist), self.hist_size)
        if (length == 0):
            return []
        else:
            return hist[-length:]
                        
    def entropy(self, spatial_probs, nonspatial_probs):
        ent = - (torch.mean(torch.sum(spatial_probs[:,0,:,:] * torch.log(spatial_probs[:,0,:,:]+self.eps_denom), dim=(1,2))) + torch.mean(torch.sum(nonspatial_probs * torch.log(nonspatial_probs+self.eps_denom), dim=1) ) )
        return ent
        
    def clip_gradients(self, clip):
        
        ### Clip the gradients of self.policy_net
        for param in self.net.parameters():
            if param.grad is None:
                continue
            #print(torch.max(param.grad.data), torch.min(param.grad.data))
            param.grad.data = param.grad.data.clamp(-clip, clip)
            
    def plot_results(self):
        plt.figure(1)
        plt.clf()
        plt.suptitle('Select-Move PPO')
        plt.title('Agent trained by Ray Sun, David Long, Michael McGuire', fontsize=7)
        plt.xlabel('Training iteration - DefeatRoaches')
        plt.ylabel('Average score')
        plt.plot(self.averages)
        plt.pause(0.001) # pause a bit so that plots are updated

def main():
    env = custom_env.MinigameEnvironment(state_modifier.graph_conv_modifier,
                                            map_name_="DefeatRoaches",
                                            render=False,
                                            step_multiplier=2)
    lr = 0.00025                       
    agent = PPOAgent(env, lr)
    agent.load_saved_model()
    agent.train()


if __name__ == "__main__":
    main()












