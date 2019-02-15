
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
    
        #print(obs.observation.available_actions)
    
        scr = obs.observation.feature_screen
        mmap = obs.observation.feature_minimap
        player = obs.observation.player
        
        proc_scr = state_modifier.preprocess_featuremap(scr, SCREEN_FEATURES, DeepMind2017Config.screen_shape)
        proc_mmap = state_modifier.preprocess_featuremap(mmap, MINIMAP_FEATURES, DeepMind2017Config.minimap_shape)
        player = state_modifier.preprocess_featuremap(player, None, False)
        

        
        return proc_scr, proc_mmap, player
        
        
        
    def preprocess_featuremap(x, features=None, out_shape=None):
    
        '''
        
        if features is None, return numpy array
        else return coo format indices, values
        
        '''
    
        if (type(features) == type(None)):
            return np.log(x.clip(0) + 1).reshape((1, len(x), 1, 1))
    
    
        (_,w,h) = x.shape
        
        #indices = [np.array([]) for i in range(4)]
        indices = np.zeros((4,1))
        values = np.array([])
        
        #preprocessed_features = np.zeros(out_shape)
        features_depth = 0
        
        for i in range(len(features)):
        
            
            name, scale, featuretype = (features[i].name, features[i].scale, features[i].type)
            
            vals = np.array(x[name])
            y_coords,x_coords = nonzero_indices = vals.nonzero()
            val_list = vals[nonzero_indices]
            
            n_curr = len(y_coords)
            zeros = np.zeros(n_curr)
            
            dim = scale
            if (featuretype == SCALAR or (featuretype == CATEGORICAL and dim == 2)):
                val_list = np.log(val_list.clip(0) + 1)
                dim = 1
                depths = features_depth * np.ones(n_curr)
            else:
                
                depths = val_list + features_depth
                val_list = np.ones(n_curr)                
            
            addition = np.stack([zeros, depths, x_coords, y_coords])
            if (i == 0):
                indices = addition
            else:
                indices = np.concatenate([indices, addition],1)
            
            values = np.append(values, val_list)
            
            features_depth += dim
            
            
            """
            if (featuretype == CATEGORICAL):
                dim = scale
                if (dim == 2):
                    dim=1
                    #addition = np.expand_dims(x[name], 0)
                    preprocessed_features[features_depth:features_depth+dim] = x[name]
                else:
                    addition = preprocessed_features[features_depth:features_depth+dim]
                    
                    vals = np.array(x[name])
                    nonzero_indices = vals.nonzero()
                    val_list = vals[nonzero_indices]
                    indices = list(nonzero_indices)
                    indices.insert(0, list(val_list))
                    addition[tuple(indices)] = 1
                    
            else:
                dim = 1
                preprocessed_features[features_depth:features_depth+dim]
                addition = np.expand_dims(np.log(x[name].clip(0) + 1), 0)
                
            #preprocessed_features[features_depth:features_depth+dim] = addition
            features_depth += dim
            """
            
                
        return indices, values.astype(np.float32)




















