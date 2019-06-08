from pysc2.agents import base_agent
from action_interface import Action, Actuator
from modified_state_space import state_modifier
from custom_env import MinigameEnvironment
import numpy as np
from scipy import ndimage
from scipy.spatial import distance


class Agent(base_agent.BaseAgent):

    def __init__(self):
        super().__init__()
        self.actuator = Actuator()
        self.reset()

    def reset(self):
        super().reset()
        self.actuator.reset()
        self._select_next = True

    def step(self, obs):
        super().step(obs)

        features = self._modified_state_space(obs)
        selected = features[0]
        friendly_unit_density = features[2]
        enemy_unit_density = features[4]
        if np.all(friendly_unit_density == 0):
            return 4,
        if self._select_next or np.all(selected == 0):
            self._select_next = False
            return 0,
        else:
            self._select_next = True
            target = self._compute_attack_closest(selected, enemy_unit_density)
            return 1, target

    @staticmethod
    def _modified_state_space(obs):
        _PLAYER_FRIENDLY = 1
        _PLAYER_HOSTILE = 4

        def zero_one_norm(array):
            arr_max = np.max(array)
            arr_min = np.min(array)
            denom = arr_max - arr_min
            if (denom == 0):
                return array
            return (array - arr_min) / denom

        scr = obs.observation.feature_screen

        # Computes array of locations of selected marines
        friendly_selected = np.array(scr.selected)

        # Computes arrays of locations of marines and enemy units
        player_relative = np.array(scr.player_relative)
        player_friendly = (player_relative == _PLAYER_FRIENDLY).astype(int)
        player_hostile = (player_relative == _PLAYER_HOSTILE).astype(int)

        # Computes arrays of hitpoints for marines and enemy units
        player_hitpoints = np.array(scr.unit_hit_points)
        friendly_hitpoints = np.multiply(player_hitpoints, player_friendly)
        hostile_hitpoints = np.multiply(player_hitpoints, player_hostile)

        # Computes arrays of density for marines and enemy units
        unit_density = np.array(scr.unit_density)
        friendly_density = np.multiply(unit_density, player_friendly)
        hostile_density = np.multiply(unit_density, player_hostile)

        # Normalize friendly_hitpoints and hostile_hitpoints to between 0 and 1
        friendly_hitpoints = zero_one_norm(friendly_hitpoints)
        hostile_hitpoints = zero_one_norm(hostile_hitpoints)

        # Stacks the previous arrays in the order given in the documentation. This will be the primary input to the neural network.

        array = np.stack([friendly_selected, friendly_hitpoints,
                          friendly_density, hostile_hitpoints, hostile_density], axis=0)

        return array

    @staticmethod
    def _compute_attack_closest(selected, enemy_unit_density):
        friendly_com = np.expand_dims(
            np.array(ndimage.measurements.center_of_mass(selected)), axis=0)
        enemy_positions = np.transpose(enemy_unit_density.nonzero())
        distances = distance.cdist(friendly_com, enemy_positions)
        closest = np.flip(enemy_positions[np.argmin(distances)], 0)
        return closest


if __name__ == "__main__":
    agent = Agent()
    env = MinigameEnvironment(state_modifier.modified_state_space,
                              map_name_="DefeatRoaches",
                              render=True,
                              step_multiplier=1)
    r = True
    while True:
        if r:
            env.reset()
            agent.reset()
        action = agent.step(env._curr_frame)
        _, _, r, _ = env.step(*action)
