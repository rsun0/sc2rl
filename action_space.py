from pysc2.lib import actions
from pysc2.lib import features
from enum import Enum
from enum import auto

class Action(Enum):
    NO_OP = auto()
    RETREAT = auto()
    ATTACK_CLOSEST = auto()
    ATTACK_WEAKEST = auto()

_SINGLE_SELECT = [0]

class Actuator:

    def __init__(self):
        # the action that the actuator should perform next
        self.state = Action.NO_OP

    def get_next_action(self, selected, enemy_unit_density, enemy_hit_points):
        '''
        Gets the raw action to fulfill the actuator's current state.

        :param selected: the selected raw feature
        :param enemy_unit_density: the enemy unit_density abstracted feature
        :param enemy_hit_points: the enemy hit_points abstracted feature
        :returns: raw action to use
        '''
        if self.state == Action.NO_OP:
            return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
        elif self.state == Action.RETREAT:
            return self._compute_retreat(selected, enemy_unit_density)
        elif self.state == Action.ATTACK_CLOSEST:
            return self._compute_attack_closest(selected, enemy_unit_density)
        elif self.state == Action.ATTACK_WEAKEST:
            return self._compute_attack_weakest(selected, enemy_hit_points)
        else:
            raise Exception('Actuator in invalid state')

    def set_order(self, action, point):
        '''
        Sets the abstracted action for the actuator to perform.
        Actuator state must be NO_OP.

        :param action: abstracted action
        :type action: Action enum
        :param point: location of unit
        :returns: raw action to use
        '''
        if self.state != Action.NO_OP:
            raise Exception('Cannot set an order for the actuator when it is already doing something')
        self.state = action
        return actions.FunctionCall(actions.FUNCTIONS.select_point.id, [_SINGLE_SELECT, point])

    def _compute_retreat(self, selected, enemy_unit_density):
        pass

    def _compute_attack_closest(self, selected, enemy_unit_density):
        pass

    def _compute_attack_weakest(self, selected, enemy_hit_points):
        pass