from pysc2.lib import actions
from pysc2.lib import features
from enum import Enum
from enum import auto
from scipy import ndimage
import numpy as np
import random

_SINGLE_SELECT = [0]
_NOT_QUEUED = [0]

class Action(Enum):
    NO_OP = auto()
    SELECT = auto()
    RETREAT = auto()
    ATTACK_CLOSEST = auto()
    ATTACK_WEAKEST = auto()

class _ActuatorState(Enum):
    NOT_SELECTED = auto()
    SELECTED = auto()

class Actuator:

    def __init__(self):
        self.state = _ActuatorState.NOT_SELECTED

    def compute_action(self, action, selected, own_unit_density, enemy_unit_density, enemy_hit_points):
        '''
        Gets the raw action to fulfill the actuator's current state.

        :param selected: the selected raw feature
        :param enemy_unit_density: the enemy unit_density abstracted feature
        :param enemy_hit_points: the enemy hit_points abstracted feature
        :returns: raw action to use
        '''
        if action == Action.NO_OP:
            return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
        
        if self.state == _ActuatorState.NOT_SELECTED:
            if action == Action.SELECT:
                self.state = _ActuatorState.SELECTED
                return self._compute_select(own_unit_density)
            raise Exception('Actuator cannot order a unit without selection or twice in a row')
        else:
            self.state = _ActuatorState.NOT_SELECTED
            if action == Action.RETREAT:
                return self._compute_retreat(selected, enemy_unit_density)
            elif action == Action.ATTACK_CLOSEST:
                return self._compute_attack_closest(selected, enemy_unit_density)
            elif action == Action.ATTACK_WEAKEST:
                return self._compute_attack_weakest(selected, enemy_hit_points)
            raise Exception('Actuator cannot select twice in a row')

    def _compute_select(self, own_unit_density):
        possible_y, possible_x = own_unit_density.nonzero()
        possible_points = zip(possible_x, possible_y)
        point = random.choice(possible_points)
        return actions.FunctionCall(actions.FUNCTIONS.select_point.id, [_SINGLE_SELECT, point])

    def _compute_retreat(self, selected, enemy_unit_density):
        enemy_com = ndimage.measurements.center_of_mass(enemy_unit_density)
        unit_position = selected.nonzero()[0][0], selected.nonzero()[1][0]
        direction_vector = -(enemy_com[1] - unit_position[1]), -(enemy_com[0] - unit_position[0])

        max_movement_x = self._compute_movement_multiple(direction_vector[0], unit_position, enemy_unit_density.shape[1])
        max_movement_y = self._compute_movement_multiple(direction_vector[1], unit_position, enemy_unit_density.shape[0])
        max_movement = min(max_movement_x, max_movement_y)
        
        retreat_target = max_movement * direction_vector
        retreat_target = round(retreat_target[0]), round(retreat_target[1])
        return actions.FunctionCall(actions.FUNCTIONS.Move_screen.id, [_NOT_QUEUED, retreat_target])

    @staticmethod
    def _compute_movement_multiple(direction, position, screen_size):
        if direction < 0:
            max_movement = position / -direction
        elif direction == 0:
            max_movement = np.infty
        else:
            max_movement = (screen_size - 1 - position) / direction
        return max_movement

    def _compute_attack_closest(self, selected, enemy_unit_density):
        pass

    def _compute_attack_weakest(self, selected, enemy_hit_points):
        pass