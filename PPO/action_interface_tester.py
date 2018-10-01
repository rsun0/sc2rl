from pysc2.agents import base_agent
from action_interface import Action, Actuator
from modified_state_space import state_modifier
import numpy as np

class Agent(base_agent.BaseAgent):

    def __init__(self):
        super().__init__()
        self.actuator = Actuator()

    def reset(self):
        super().reset()
        self.actuator.reset()

    def step(self, obs):
        super().step(obs)

        features, _ = state_modifier.modified_state_space(obs)
        selected = features[0]
        friendly_unit_density = features[2]
        enemy_unit_density = features[4]
        if np.all(friendly_unit_density == 0):
            return self.actuator.compute_action(Action.NO_OP, selected, friendly_unit_density, enemy_unit_density)
        if not self.actuator.units_selected or np.all(selected == 0):
            return self.actuator.compute_action(Action.SELECT, selected, friendly_unit_density, enemy_unit_density)
        else:
            return self.actuator.compute_action(Action.ATTACK, selected, friendly_unit_density, enemy_unit_density)