
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
from pysc2.lib import features
from pysc2.env import environment

import numpy as np

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_HITPOINTS = features.SCREEN_FEATURES.unit_hit_points.index
_PLAYER_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_PLAYER_UNIT_DENSITY = features.SCREEN_FEATURES.unit_density.index
_SELECTED = features.SCREEN_FEATURES.selected.index
_ARMY_COUNT = 8
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]
_FIRST_TIMESTEP = environment.StepType.FIRST
_LAST_TIMESTEP = environment.StepType.LAST


class state_modifier():        
    
    
    
    """
        inputs : TimeStep variable for each frame's observation

        outputs : a stacked 3d numpy tensor stacked as
        
            Current position of marines
            Hit points of all marines
            Unit density of all marines
            Hit points of all roaches
            Unit density of all roaches
            

    """
    def modified_state_space(obs):
    
        scr = obs.observation.feature_screen
   
        ### Computes array of locations of selected marines
        friendly_selected = np.array(scr.selected)
    
        ### Computes arrays of locations of marines and enemy units
        player_relative = np.array(scr.player_relative)
        player_friendly = (player_relative == _PLAYER_FRIENDLY).astype(int)
        player_hostile = (player_relative == _PLAYER_HOSTILE).astype(int)
        
        ### Computes arrays of hitpoints for marines and enemy units
        player_hitpoints = np.array(scr.unit_hit_points) 
        friendly_hitpoints = np.multiply(player_hitpoints, player_friendly)
        hostile_hitpoints = np.multiply(player_hitpoints, player_hostile)
        
        ### Computes arrays of density for marines and enemy units
        unit_density = np.array(scr.unit_density)
        friendly_density = np.multiply(unit_density, player_friendly)
        hostile_density = np.multiply(unit_density, player_hostile) 
        
        ### Stacks the previous arrays in the order given in the documentation. This will be the primary input to the neural network.
        array = np.stack([friendly_selected, friendly_hitpoints, friendly_density, hostile_hitpoints, hostile_density], axis=0)
        
        # ------------------------------------- #        
        
        ### Computes the number of marines that are still alive
        # army_count = len(obs.observation.feature_units)
        
        # return (array, army_count)
        return array
        
        
        

       























