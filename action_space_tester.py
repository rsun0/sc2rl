from pysc2.agents import base_agent
from action_space import Action, Actuator
from modified_state_space import state_modifier
import numpy as np

class Agent(base_agent.BaseAgent):

    def __init__(self):
        super().__init__()
        self.actuator = Actuator()
        self.state = 0
        self.state_space = state_modifier()

    def reset(self):
        super().reset()
        self.actuator = Actuator()
        self.state = 0

    def step(self, obs):
        super().step(obs)

        features, _, _ = state_modifier.modified_state_space(obs)
        selected = features[0]
        friendly_unit_density = features[3]
        enemy_unit_density = features[4]
        enemy_hit_points = features[2]
        if np.all(friendly_unit_density == 0):
            return self.actuator.compute_action(Action.NO_OP, selected, friendly_unit_density, enemy_unit_density, enemy_hit_points)
        if self.state == 0 or np.all(selected == 0):
            self.state = 1
            return self.actuator.compute_action(Action.SELECT, selected, friendly_unit_density, enemy_unit_density, enemy_hit_points)
        else:
            self.state = 0
            return self.actuator.compute_action(Action.ATTACK_CLOSEST, selected, friendly_unit_density, enemy_unit_density, enemy_hit_points)