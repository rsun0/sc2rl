from agent import Agent, Model, Memory, AgentSettings
from config import GraphConvConfigMinigames
from modified_state_state import state_modifier
import utils

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import copy

class PPOAgent(Agent):
    
    def __init__(self, model, settings, memory, PPO_settings):
        super().__init__(model, settings, memory)
        self.step = 0
        self.frame_count = 0
        self.epochs_trained = 0
        self.PPO_settings = PPO_settings
        self.target_model = copy.deepcopy(model)
        self.hidden_state = self.model.init_hidden(use_torch=False)
        self.prev_hidden_state = None
        self.optimizer = settings.optimizer(self.model.parameters(),
                                            lr=settings.learning_rate)
        self.action = [np.array([[0,0],[0,0]]), 0]
        self.config = GraphConvConfigMinigames
        self.value = 0
        self.loss = nn.MSELoss().to(PPO_settings['device'])
        
    def _forward(self, agent_state, choosing=False, mode=self.config.SELECT):
        (G, X, avail_actions) = agent_state
        self.prev_action = utils.action_to_onehot(self.action,
                                                    self.config.action_space,
                                                    self.config.spatial_width)
        self.prev_hidden_state = copy.deepcopy(self.hidden_state)
        _, _, value, self.hidden_state, action = self.model(np.expand_dims(G, 1),
        
                                            np.expand_dims(X, 1),
                                            avail_actions,
                                            self.hidden_state,
                                            np.expand_dims(self.prev_action, 1)
                                epsilon=self.settings.get_epsilon(self.frame_count),
                                            choosing=True,
                                            mode=mode)
                                            
        return _, _, value.cpu().data.numpy().item(), self.hidden_state.cpu().data.numpy(), action
        
        
    def _sample(self, agent_state):
    
        mode = self.config.SELECT
        if (self.step % 2 == 1):
            mode = ATTACK
        _, _, self.value, self.hidden_state, self.action = self._forward(agent_state,
                                                                    choosing=True,
                                                                    mode=mode)
        self.frame_count += 1
        self.step += 1
        return self.action
        
    def state_space_converter(self, state):
        return state_modifier.graph_conv_modifier(state)[:3]
        
    def action_space_converter(self, personal_action):
        return personal_action
        
    def train(self, run_settings):
        self.memory.compute_vtargets_adv(self.PPO_settings['discount_factor'],
                                            self.PPO_settings['lambda'])
        batch_size = run_settings.batch_size
        num_iters = int(len(self.memory) / batch_size)
        epochs = run_settings.num_epochs
        
        for i in range(epochs):
            
            pol_loss = 0
            vf_loss = 0
            ent_total =0
            
            for j in range(num_iters):
                
                d_pol, d_vf, d_ent = self.train_step(batch_size)
                pol_loss += d_pol
                
            self.epochs_trained += 1
            pol_loss /= num_iters
            vf_loss /= num_iters
            ent_total /= num_iters
            print("Epoch %d: Policy loss: %f. Value loss: %f. Entropy %f" % 
                            (self.epochs_trained, 
                            pol_loss, 
                            vf_loss, 
                            ent_total)
                            )
        self.update_target_net()
        
        print("\n\n ------- Training sequence ended ------- \n\n")
        
    def train_step(self, batch_size):
        
        




























