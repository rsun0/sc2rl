from agent import Agent, Model, Memory, AgentSettings

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import copy


class RRLAgent(Agent):

    def __init__(self, model, settings, memory, agent_settings):
        super().__init__(model, settings, memory)
        self.step = 0
        self.frame_count = 0
        self.epochs_trained = 0
        self.agent_settings = agent_settings
        self.target_model = copy.deepcopy(model)
        self.hidden_state = self.model.init_hidden(use_torch=False)
        self.prev_hidden_state = None
        self.action = [0, np.zeros(10), np.zeros((3,2))]

    def _forward(self, agent_state, choosing=True):
        (minimap, screen, player, avail_actions) = agent_state
        self.prev_hidden_state = copy.deepcopy(self.hidden_state)
        _, _, _, self.hidden_state, value, action = self.model(minimap,
                                                            screen,
                                                            player,
                                                            avail_actions,
                                                            np.array([self.action[0]]),
                                                            self.hidden_state,
                                                            choosing=choosing)
        return _, _, value.cpu().data.numpy().item(), self.hidden_state.cpu().data.numpy(), action

    def _sample(self, agent_state):
        _, _, self.value, self.hidden_state, self.action = self._forward(agent_state, choosing=True)
        self.frame_count += 1
        self.step += 1
        return self.action

    def state_space_converter(self, state):
        return state

    def action_space_converter(self, personal_action):
        return personal_action

    def train(self, run_settings):
        raise NotImplementedError

    def train_step(self, batch_size):
        raise NotImplementedError

    def load(self):
        self.net.load_state_dict(torch.load("save_model/Starcraft2" + self.env.map + "RRL.pth"))
        self.update_target_net()

    def save(self):
        torch.save(self.model.state_dict(), "save_model/Starcraft2" + self.env.map + "RRL.pth")

    ### Unique RRL functions below this line

    def update_target_net(self):
        self.target_model.load_state_dict(self.model.state_dict())
