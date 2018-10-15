
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
import random
import enhancedbaseagent
from enhancedbaseagent import EnhancedBaseAgent
from modified_state_space import state_modifier
<<<<<<< HEAD
from action_interface import Action, Actuator
from ppo_agent import PPOAgent
=======
import numpy as np
>>>>>>> 50eda19452d03e2520e35a9c714c6e0efadc1a30

class terran_agent(EnhancedBaseAgent):



    def __init__(self):
        super(terran_agent, self).__init__()
        self.attack_coordinates = None
        self.iteration = 0
        
        self.actuator = Actuator()
        self.PPO = PPOAgent()
        
    
        
    def step(self, obs):
<<<<<<< HEAD
=======
    
        simp_obs = state_modifier.modified_state_space(obs)
    
        self.iteration += 1
>>>>>>> 50eda19452d03e2520e35a9c714c6e0efadc1a30
        super(terran_agent, self).step(obs)
        self.iteration += 1
        ###################################################
        
        
        
        simp_obs = state_modifier.modified_state_space(obs)
        selected = simp_obs[0]
        friendly_unit_density = simp_obs[2]
        enemy_unit_density = simp_obs[4]
        
        
        #### Compute action using network - change later
        action = random.randint(0, 1)
        ####
        
        
        if np.all(friendly_unit_density == 0):
            return self.actuator.compute_action(Action.NO_OP, selected, friendly_unit_density, enemy_unit_density)
        if not self.actuator.units_selected or np.all(selected == 0):
            return self.actuator.compute_action(Action.SELECT, selected, friendly_unit_density, enemy_unit_density)
        elif (action == 0):
            return self.actuator.compute_action(Action.RETREAT, selected, friendly_unit_density, enemy_unit_density)
        elif (action == 1):
            return self.actuator.compute_action(Action.ATTACK, selected, friendly_unit_density, enemy_unit_density)
        
        
        """
        if self.unit_type_is_selected(obs, units.Terran.Marine) and self.iteration % 20 == 0:
            return self.handle_action(obs)
            
        else:
            return self.select_units_by_type(obs, units.Terran.Marine)
        """
        
        return actions.FUNCTIONS.no_op()
        
        
        
        
    # Implement this     
    def handle_action(self, 
                        obs):
        x = random.randint(0,83)
        y = random.randint(0,83)
        return actions.FUNCTIONS.Attack_screen("now", (x,y))



def run_game_with_agent(agent, mapname, iterations):
    game_data = []
    with sc2_env.SC2Env(
        map_name=mapname,
        agent_interface_format=features.AgentInterfaceFormat(
            feature_dimensions=features.Dimensions(screen=84, minimap=64),
            use_feature_units=True),
        step_mul=1,
        visualize=True,
        game_steps_per_episode=0) as env:
                
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


def main(unused_argv):
    agent = terran_agent()
    try:
        DZaB_data = run_game_with_agent(agent, "DefeatZerglingsAndBanelings", 10)
        DR_data = run_game_with_agent(agent, "DefeatRoaches", 10)
                    
    except KeyboardInterrupt:
        pass

        
if __name__ == "__main__":
    app.run(main)
