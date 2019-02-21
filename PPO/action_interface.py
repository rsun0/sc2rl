from pysc2.lib import actions
from enum import Enum
import numpy as np


class Action(Enum):
    SELECT = 1
    ATTACK = 2
    MOVE = 3
    STOP = 4
    NO_OP = 5


class Actuator:

    _ACTION_SPACE = (5, 2)
    _SELECT_SPACE = 84
    _SCREEN = 84

    def __init__(self):
        self.reset()

    def reset(self):
        self.units_selected = False
        self._select_index = 0

    def compute_action(self, action, raw_obs, topleft=None, botright=None):
        '''
        Computes the raw action corresponding to the chosen abstract action
        :param action: The chosen abstract action (NO_OP, SELECT, RETREAT, or ATTACK)
        :param custom_obs: Custom observations as given by modified state space
        :param raw_obs: Observations from pysc2
        :returns: The raw action to return to environment
        '''
        if action == Action.SELECT.value:
            return Actuator._select(topleft, botright)

        if action == Action.ATTACK.value:
            return actions.FUNCTIONS.Attack_screen('now', topleft)

        if action == Action.MOVE.value:
            return actions.FUNCTIONS.Move_screen('now', topleft)

        if action == Action.STOP.value:
            return actions.FUNCTIONS.Stop_quick('now')

        if action == Action.NO_OP.value:
            return actions.FUNCTIONS.no_op()

        return actions.FUNCTIONS.no_op()

    @staticmethod
    def _screen_normalize(coords):
        coords = coords.reshape((2,))
        coords = np.clip(coords, 0, Actuator._SCREEN - 1)
        return coords

    @staticmethod
    def _select(topleft, botright):

        tl = np.array(topleft) * (Actuator._SCREEN /
                                  (Actuator._SELECT_SPACE-1))
        br = np.array(botright) * (Actuator._SCREEN /
                                   (Actuator._SELECT_SPACE-1))

        tl_transform = Actuator._screen_normalize(tl)
        br_transform = Actuator._screen_normalize(br)

        return actions.FUNCTIONS.select_rect('select', tl_transform, br_transform)
