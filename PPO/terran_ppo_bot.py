
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
import random

class terran_agent(base_agent.BaseAgent):
    def __init__(self):
        super(terran_agent, self).__init__()
        self.attack_coordinates = None
        self.iteration = 0
        
    def step(self, obs):
        super(terran_agent, self).step(obs)
        
        if self.unit_type_is_selected(obs, units.Terran.Marine):
            self.handle_action(obs)
        else:
            self.handle_select(obs)
            
        self.iteration += 1
        return actions.FUNCTIONS.no_op()
        
    # Implement this     
    def handle_action(self, 
                        obs):
        return actions.FUNCTIONS.no_op()
        
    # Selects all marines
    def handle_select(self, 
                        obs):
        marines = self.get_units_by_type(obs, units.Terran.Marine)
        if (len(marines) > 0):
            marine = random.choice(marines)
            return actions.FUNCTIONS.select_point("select_all_type", (larva.x,
                                                                      larva.y))
                                                                      
                                                                      
    def unit_type_is_selected(self, obs, unit_type):
        if (len(obs.observation.single_select) > 0 and
          obs.observation.single_select[0].unit_type == unit_type):
            return True
    
        if (len(obs.observation.multi_select) > 0 and
          obs.observation.multi_select[0].unit_type == unit_type):
            return True
    
        return False






STEP_MUL=1

def main(unused_argv):
    agent = terran_agent()
    try:
        while True:
            with sc2_env.SC2Env(
                map_name="DefeatZerglingsAndBanelings",
                agent_interface_format=features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(screen=84, minimap=64),
                    use_feature_units=True),
                step_mul=8,
                visualize=True,
                game_steps_per_episode=0) as env:
                
                agent.setup(env.observation_spec(), env.action_spec())
                timesteps = env.reset()
                agent.reset()
                
                while True:
                    step_actions = [agent.step(timesteps[0])]
                    if timesteps[0].last():
                        break
                    
    except KeyboardInterrupt:
        pass
        
if __name__ == "__main__":
    app.run(main)
