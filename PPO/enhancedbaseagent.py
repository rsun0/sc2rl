
"""
Written by Michael McGuire to provide an enhanced base agent with simple repetitive functions. (09 / 29 / 2018)
"""

import pysc2
from pysc2.lib import actions, features, units
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from absl import app
import random

class EnhancedBaseAgent(base_agent.BaseAgent):

    def __init__(self):
        super(EnhancedBaseAgent, self).__init__()
    
    
    
    """
        Returns True if obs shows that there is a unit of type unit_type currently selected
    """
    def unit_type_is_selected(self, obs, unit_type):
        
        if (len(obs.observation.single_select) > 0 and
          obs.observation.single_select[0].unit_type == unit_type):
            return True
    
        if (len(obs.observation.multi_select) > 0 and
          obs.observation.multi_select[0].unit_type == unit_type):
            return True
    
        return False
        
        
        
    """
        Selects all units of type unit_type 
    """
    def select_units_by_type(self, obs, unit_type):
        
        my_units = self.get_units_by_type(obs, unit_type)
        if (len(my_units) > 0):
            u = random.choice(my_units)
            return actions.FUNCTIONS.select_point("select_all_type", (u.x, u.y))



    """
        Returns list of units of type unit_type in the feature_units layer of obs
    """
    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units if unit.unit_type == unit_type]

"""
Runs a given agent on the map mapname. Runs the agent on the mapname for i in iterations.
"""
def run_game_with_agent(agent, mapname, iterations):
    game_data = []
    env = sc2_env.SC2Env(
        map_name=mapname,
        agent_interface_format=features.AgentInterfaceFormat(
            feature_dimensions=features.Dimensions(screen=84, minimap=64),
            use_feature_units=True),
        step_mul=1,
        visualize=True,
        game_steps_per_episode=0)
        
    for i in range(1):
        agent.setup(env.observation_spec(), env.action_spec())
        
        
        for i in range(iterations):
            print("Playing game {}".format(i+1))
            timesteps = env.reset()
            agent.reset()
                    
            while True:
                step_actions = [agent.step(timesteps[0])]
                if timesteps[0].last():
                    game_data.append(timesteps)
                    break
                timesteps = env.step(step_actions)
    return game_data                
                

