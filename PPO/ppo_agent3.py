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

np.set_printoptions(linewidth=200, precision=4)


class Network(object):
    def __init__(self, env, scope, screen_plc, minimap_plc, nonspatial_act_plc, spatial_act_plc, trainable=True):
        
        self.filters1 = 16
        self.filters2 = 32
        self.filters3 = 64
        
        
        self.env = env
        self.scope = scope
        
        self.screen_plc = screen_plc
        self.minimap_plc = minimap_plc
        self.nonspatial_act_plc = nonspatial_act_plc
        self.spatial_act_plc = spatial_act_plc
        
        self.network = self._build_DeepMind2017_network()
        
        self.spatial_p = self.network.forward_spatial
        self.nonspatial_p = self.network.forward_nonspatial
        self.v = self.network.forward_v
        
        
    def _build_Deepmind2017_network(self):
        
