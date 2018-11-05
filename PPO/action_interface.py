from pysc2.lib import actions
from enum import Enum
import numpy as np
from scipy import ndimage
from scipy.spatial import distance

class Action(Enum):
    NO_OP = 1
    SELECT = 2
    LEFT = 3
    UP_LEFT = 4
    UP = 5
    UP_RIGHT = 6
    RIGHT = 7
    DOWN_RIGHT = 8
    DOWN = 9
    DOWN_LEFT = 10
    ATTACK_CLOSEST = 11
    ATTACK_WEAKEST = 12

class Actuator:
    _MOVE_MULTIPLIER = 10

    def __init__(self):
        self.reset()

    def reset(self):
        self.units_selected = False
        self._select_index = 0

    def compute_action(self, action, custom_obs, raw_obs, topleft=None, botright=None):
        '''
        Computes the raw action corresponding to the chosen abstract action
        :param action: The chosen abstract action (NO_OP, SELECT, RETREAT, or ATTACK)
        :param custom_obs: Custom observations as given by modified state space
        :param raw_obs: Observations from pysc2
        :returns: The raw action to return to environment
        '''
        if action == Action.NO_OP:
            return actions.FUNCTIONS.no_op()

        selected = custom_obs[0]
        friendly_unit_density = custom_obs[2]
        enemy_hit_points = custom_obs[3]
        enemy_unit_density = custom_obs[4]

        if np.all(selected == 0):
            self.units_selected = False

        if (topleft != None):
            assert action == Action.SELECT
            return self._select(topleft, botright)

        if not self.units_selected:
            assert action == Action.SELECT, 'Actuator cannot order units without selection (unit may have died)'

            self.units_selected = True
            return self._compute_select(friendly_unit_density, len(raw_obs.observation.feature_units))
        
        else:
            self.units_selected = False
            if action == Action.LEFT:
                return self._compute_move(selected,
                                            -1,
                                            0)
            elif action == Action.UP_LEFT:
                return self._compute_move(selected,
                                            -1,
                                            1)
            elif action == Action.UP:
                return self._compute_move(selected,
                                            0,
                                            1)
            elif action == Action.UP_RIGHT:
                return self._compute_move(selected,
                                            1,
                                            1)
            elif action == Action.RIGHT:
                return self._compute_move(selected,
                                            1,
                                            0)
            elif action == Action.DOWN_RIGHT:
                return self._compute_move(selected,
                                            1,
                                            -1)
            elif action == Action.DOWN:
                return self._compute_move(selected,
                                            0,
                                            -1)
            elif action == Action.DOWN_LEFT:
                return self._compute_move(selected,
                                            -1,
                                            -1)
            elif action == Action.ATTACK_CLOSEST:
                return self._compute_attack_closest(selected, enemy_unit_density)
            elif action == Action.ATTACK_WEAKEST:
                return self._compute_attack_weakest(selected, enemy_unit_density, enemy_hit_points)
            elif action == Action.NO_OP:
                return actions.FUNCTIONS.no_op()
            assert False, 'Invalid action passed (actuator cannot select with preexisting selection)'

    @staticmethod
    def _compute_retreat(selected, enemy_unit_density):
        friendly_com = np.flip(np.array(ndimage.measurements.center_of_mass(selected)), 0)
        enemy_com = np.flip(np.array(ndimage.measurements.center_of_mass(enemy_unit_density)), 0)
        direction_vector = -(enemy_com - friendly_com)

        max_movement_x = Actuator._compute_movement_multiplier(direction_vector[0], friendly_com[0], selected.shape[1])
        max_movement_y = Actuator._compute_movement_multiplier(direction_vector[1], friendly_com[1], selected.shape[0])
        max_movement = min(max_movement_x, max_movement_y)
        
        retreat_target = np.round(max_movement * direction_vector + friendly_com)
        return actions.FUNCTIONS.Move_screen('now', retreat_target)

    @staticmethod
    def _compute_movement_multiplier(direction, position, screen_size):
        if direction < 0:
            return position / -direction
        elif direction == 0:
            return np.infty
        else:
            return (screen_size - 1 - position) / direction

    # @staticmethod
    # def _compute_attack(enemy_unit_density):
    #     enemy_com = np.flip(np.array(ndimage.measurements.center_of_mass(enemy_unit_density)), 0)
    #     return actions.FUNCTIONS.Attack_screen('now', enemy_com)

    def _compute_select(self, friendly_unit_density, num_units):
        # return actions.FUNCTIONS.select_army('select')
        possible_points = np.transpose(np.nonzero(friendly_unit_density))
        if len(possible_points) == 0 or num_units == 0:
            raise Exception('Actuator cannot select when no units exist')
        if self._select_index >= num_units:
            self._select_index = 0
        idx = int((self._select_index / num_units) * len(possible_points))
        selection = np.flip(possible_points[idx], 0)
        self._select_index += 1
        return actions.FUNCTIONS.select_point('select', selection)

    @staticmethod
    def _compute_attack_closest(selected, enemy_unit_density):
        friendly_com = np.expand_dims(np.array(ndimage.measurements.center_of_mass(selected)), axis=0)
        enemy_positions = np.transpose(enemy_unit_density.nonzero())
        distances = distance.cdist(friendly_com, enemy_positions)
        closest = np.flip(enemy_positions[np.argmin(distances)], 0)
        return actions.FUNCTIONS.Attack_screen('now', closest)

    @staticmethod
    def _compute_attack_weakest(selected, enemy_unit_density, enemy_hit_points):
        enemy_hit_points[enemy_unit_density == 0] = np.finfo(enemy_hit_points.dtype).max
        weakest = np.flip(np.array(np.unravel_index(np.argmin(enemy_hit_points), enemy_hit_points.shape)), axis=0)
        return actions.FUNCTIONS.Attack_screen('now', weakest)

    @staticmethod
    def _screen_normalize(coords, screen_size):
        coords = coords.reshape((2,))
        for i in range(len(coords)):
            if coords[i] < 0:
                coords[i] = 0
            if coords[i] > screen_size - 1:
                coords[i] = screen_size - 1
        return coords

    @staticmethod
    def _compute_move(selected, dx, dy):
        friendly_com = np.expand_dims(np.array(ndimage.measurements.center_of_mass(selected)), axis=0)
        direction_vec = Actuator._MOVE_MULTIPLIER * np.array([dy, dx])
        move_target = Actuator._screen_normalize(friendly_com + direction_vec, 84)
        return actions.FUNCTIONS.Move_screen('now', move_target)
        
    @staticmethod
    def _select(topleft, botright):
        return actions.FUNCTIONS.select_rect(topleft, botright)
