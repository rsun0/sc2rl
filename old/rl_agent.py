from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from modified_state_space import state_modifier

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.env import environment



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


class DefeatRoaches(base_agent.BaseAgent):

    def __init__(self):
        self.replay_mem = []
        self.policy = Net()
        env_handler = EnvironmentHandler()


    def step(self, obs):
        
        reduced_obs = state_modifier.modified_state_space(obs)
        sarsd = env_handler.step(reduced_obs, self.policy)
        
        if (game_finished(sarsd)):
            self.train()
            
