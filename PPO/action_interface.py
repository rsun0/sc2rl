from pysc2.lib import actions
from enum import Enum
from enum import auto
import numpy as np
from scipy import ndimage

class Action(Enum):
    NO_OP = auto()
    SELECT = auto()
    RETREAT = auto()
    ATTACK = auto()

class Actuator:
    def __init__(self):
        self.units_selected = False

    def reset(self):
        self.units_selected = False

    def compute_action(self, action, selected, friendly_unit_density, enemy_unit_density):
        '''
        Computes the raw action corresponding to the chosen abstract action

        :param action: The chosen abstract action (NO_OP, SELECT, RETREAT, or ATTACK)
        :param selected: 
        :param friendly_unit_density: The abstract feature representing friendly unit locations
        :param enemy_unit_density: The abstract feature representing enemy unit locations
        :returns: The raw action to return to environment
        '''
        if action == Action.NO_OP:
            return actions.FUNCTIONS.no_op()

        if np.all(selected == 0):
            self.units_selected = False

        if not self.units_selected:
            if action == Action.SELECT:
                self.units_selected = True
                return actions.FUNCTIONS.select_army('select')
            raise Exception('Actuator cannot order units without selection (unit may have died)')
        
        else:
            if action == Action.RETREAT:
                return self._compute_retreat(friendly_unit_density, enemy_unit_density)
            elif action == Action.ATTACK:
                return self._compute_attack(enemy_unit_density)
            raise Exception('Actuator cannot select with preexisting selection')

    @staticmethod
    def _compute_retreat(friendly_unit_density, enemy_unit_density):
        friendly_com = np.flip(np.array(ndimage.measurements.center_of_mass(friendly_unit_density)), 0)
        enemy_com = np.flip(np.array(ndimage.measurements.center_of_mass(enemy_unit_density)), 0)
        direction_vector = -(enemy_com - friendly_com)

        max_movement_x = Actuator._compute_movement_multiplier(direction_vector[0], friendly_com[0], friendly_unit_density.shape[1])
        max_movement_y = Actuator._compute_movement_multiplier(direction_vector[1], friendly_com[1], friendly_unit_density.shape[0])
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

    @staticmethod
    def _compute_attack(enemy_unit_density):
        enemy_com = np.flip(np.array(ndimage.measurements.center_of_mass(enemy_unit_density)), 0)
        return actions.FUNCTIONS.Attack_screen('now', enemy_com)