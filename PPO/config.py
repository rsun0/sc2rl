"""

Put relevant constants in this file

"""
from pysc2.lib.features import SCREEN_FEATURES, MINIMAP_FEATURES, FeatureType
CATEGORICAL = FeatureType.CATEGORICAL
SCALAR = FeatureType.SCALAR
import numpy as np

screen_shape = (1968,84,84)
minimap_shape = (30,84,84)
nonspatial_size = 11
mid_screen = 42.
screen_size = 84.

class DeepMind2017Config():
    screen_shape = (1968,84,84)
    minimap_shape = (30,84,84)
    nonspatial_size = 11
    FILTERS1 = 16
    FILTERS2 = 32
    FC1SIZE = 256
    FC2SIZE = 256
    latent_size = 50
    
    action_space = 5
    spatial_action_space = (2,84,84)
    env_agent_action_mapper = {0:4,
                                3:0,
                                12:1,
                                331:2,
                                453:3}
    
class GraphConvConfigMinigames():
    ### Unit id codes
    Marine = 48
    Roach = 110
    
    ### Alliance id codes
    friendly = 1
    enemy = 4
    k_player = 3
    k_opponent = 2
    
    index_dict = {Marine: 0,
                    Roach: 1}
                    
    num_unit_types = len(index_dict)
    num_alliances = 5
    num_directions = 8
    
    categorical_size_dict = {0: num_unit_types,     # Convert to one-hot for unit categories
                                1: num_alliances,   # Number of alliances
                                14: num_directions  # direction unit is facing
                                }
                                
    loc_idx = x_ind, y_ind = [12, 13]
    
    unit_vec_width = 27 + (num_unit_types-1) + (num_alliances - 1) + (num_directions - 1)
    
    # Uniform for all 
    graph_n = 20
    
    # env configs
    action_space = 5
    
    env_agent_action_mapper = {0:4,
                                3:0,
                                12:1,
                                331:2,
                                453:3}
                                

    spatial_width = 32    

