
"""

This file contains modified_state_space(obs), a function for generating a modified,
simplified state space for the pysc2 learning environment.


Author : Michael McGuire
Date : 02/14/2018
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features, units
from pysc2.env import environment


from config import *


import numpy as np
def zero_one_norm(array):
    arr_max = np.max(array)
    arr_min = np.min(array)
    denom = arr_max - arr_min
    if (denom == 0):
        return array
    return (array - arr_min) / denom

class state_modifier():        


    def modified_state_space(obs):
        
        '''
            IN: obs
            OUT: preprocessed values for
                screen (84 x 84 x 17)
                minimap (84 x 84 x 7)    
        '''
        #print(obs.observation.feature_minimap._index_names, obs.observation.feature_screen._index_names, obs.observation.feature_minimap.shape)
        
        #print(obs.observation.feature_screen.shape, obs.observation.feature_minimap.shape, len(obs.observation.player))
    
        scr = obs.observation.feature_screen
        mmap = obs.observation.feature_minimap
        player = obs.observation.player
        
        proc_scr = state_modifier.preprocess_featuremap(scr, SCREEN_FEATURES, DeepMind2017Config.screen_shape)
        proc_mmap = state_modifier.preprocess_featuremap(mmap, MINIMAP_FEATURES, DeepMind2017Config.minimap_shape)
        player = state_modifier.preprocess_featuremap(player, None, False)
        
        print(proc_scr.shape, proc_mmap.shape, player.shape)
        
        return proc_scr, proc_mmap, player
        
        
        
    def preprocess_featuremap(x, features=None, out_shape=None):
    
        if (type(features) == type(None)):
            return np.log(x.clip(0) + 1).reshape((1, len(x), 1, 1))
    
    
        (_,w,h) = x.shape
        preprocessed_features = np.zeros(out_shape)
        features_depth = 0
        
        for i in range(len(features)):
            name, scale, featuretype = (features[i].name, features[i].scale, features[i].type)
            
            if (featuretype == CATEGORICAL):
                dim = scale
                if (dim == 2):
                    dim=1
                    addition = np.expand_dims(x[name], 0)
                else:
                    print(dim, len(x[name].nonzero()[0]))
                    addition = np.zeros((dim,h,w))
                    vals = x[name]
                    layer_idx = np.arange(h).reshape(h,1)
                    col_idx = np.tile(np.arange(w), (h,1))
                    #print(addition.shape, vals.max(), vals.min(), dim)
                    addition[vals, layer_idx, col_idx] = 1
                    #addition = (np.arange(dim) == (x[name])[...,None]).astype(int)
            else:
                dim = 1
                addition = np.expand_dims(np.log(x[name].clip(0) + 1), 0)
                
            preprocessed_features[features_depth:features_depth+dim] = addition
            features_depth += dim
            
                
        #preprocessed_features = np.swapaxes(np.swapaxes(preprocessed_features, 0, 1), 1, 2)
                
        return np.expand_dims(preprocessed_features, 0)




















