
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
from scipy.spatial.distance import cdist


import numpy as np
def zero_one_norm(array):
    arr_max = np.max(array)
    arr_min = np.min(array)
    denom = arr_max - arr_min
    if (denom == 0):
        return array
    return (array - arr_min) / denom

class state_modifier():        

    def graph_conv_modifier(obs):
        
        units = np.array(obs.observation.feature_units)
        G = state_modifier.to_graph(units)
        X = state_modifier.preprocess_units(units)
        avail_actions = state_modifier.preprocess_actions(obs.observation.available_actions, GraphConvConfigMinigames)
        
        return [G, X, avail_actions]


    def modified_state_space(obs):
        
        '''
            IN: obs
            OUT: preprocessed values for
                screen (84 x 84 x 17)
                minimap (84 x 84 x 7)    
        '''
        
        scr = obs.observation.feature_screen
        mmap = obs.observation.feature_minimap
        player = obs.observation.player
        
        proc_scr = state_modifier.preprocess_featuremap(scr, SCREEN_FEATURES, DeepMind2017Config.screen_shape)
        proc_mmap = state_modifier.preprocess_featuremap(mmap, MINIMAP_FEATURES, DeepMind2017Config.minimap_shape)
        player = state_modifier.preprocess_featuremap(player, None, False)
        
        avail_actions = state_modifier.preprocess_actions(obs.observation.available_actions, DeepMind2017Config)

        
        return np.array([proc_scr, proc_mmap, player, avail_actions])
        
    def preprocess_units(units):
    
        (n,d) = units.shape
        output = np.zeros((n, GraphConvConfigMinigames.unit_vec_width))
        key_check = GraphConvConfigMinigames.categorical_size_dict.keys()
        idx = 0
        for j in range(d):
            # Does j correspond to categorical feature?
            if j in key_check:
                # Is j a unit id index?
                if j == 0:
                    units[:,j+idx] = np.vectorize(GraphConvConfigMinigames.index_dict.get)(units[:,j])
                width = GraphConvConfigMinigames.categorical_size_dict[j]
                one_hot_mat = np.zeros((n,width))
                one_hot_mat[range(n), units[:,j]] = 1
                output[:,j+idx:j+idx+width] = one_hot_mat
                idx += (width-1)
            else:
                # Does j correpsond to an x or y coordinate?
                if (j == GraphConvConfigMinigames.x_ind or j == GraphConvConfigMinigames.y_ind):
                    output[:,j+idx] = (units[:,j] - mid_screen) / screen_size
                else:
                    output[:,j+idx] = np.log(units[:,j]+1)
        return output
                
        
    def to_graph(units):
        (n, d) = units.shape
        
        output = np.zeros((GraphConvConfigMinigames.graph_n,GraphConvConfigMinigames.graph_n))
        friendly = (units[:,1] == GraphConvConfigMinigames.friendly).astype(np.int).nonzero()[0]
        enemy = (units[:,1] == GraphConvConfigMinigames.enemy).astype(np.int).nonzero()[0]
        
        k1 = min(GraphConvConfigMinigames.k_player, max(0,len(friendly)-1))
        k2 = min(GraphConvConfigMinigames.k_opponent, max(0,len(enemy)-1))
        
        locs = units[:,GraphConvConfigMinigames.loc_idx]
        dist_array = cdist(locs, locs)
        
        m1 = dist_array[friendly]
        m2 = dist_array[enemy]
        friendly_player = m1[:,friendly]
        friendly_opponent = m1[:,enemy]
        enemy_player = m2[:,enemy]
        enemy_opponent = m2[:,friendly]
        
        fp_idx = friendly[np.argpartition(friendly_player, k1-1, axis=1)[:,:k1]]
        fo_idx = enemy[np.argpartition(friendly_opponent, k2-1, axis=1)[:,:k2]]
        ep_idx = enemy[np.argpartition(enemy_player, k2-1, axis=1)[:,:k2]]
        eo_idx = friendly[np.argpartition(enemy_opponent, k1-1, axis=1)[:,:k1]]
        
        
        output[np.expand_dims(friendly,1), fp_idx] = 1.0
        output[np.expand_dims(friendly,1), fo_idx] = 1.0
        output[np.expand_dims(enemy, 1), ep_idx] = 1.0
        output[np.expand_dims(enemy, 1), eo_idx] = 1.0
        
        return output
        
    def preprocess_actions(actions, my_config):
        available = np.zeros((my_config.action_space))
        for i in actions:
            if i in my_config.env_agent_action_mapper.keys():
                available[my_config.env_agent_action_mapper[i]] = 1
        return available
    """
    def preprocess_actions(actions):
        available = np.zeros((1,DeepMind2017Config.action_space))
        for i in actions:
            if i in DeepMind2017Config.env_agent_action_mapper.keys():
                available[0,DeepMind2017Config.env_agent_action_mapper[i]] = 1
        return available
    """    
        
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
            
                
        return [indices, values.astype(np.float32)]




















