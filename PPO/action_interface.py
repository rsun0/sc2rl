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
        self._select_index = 0

    def compute_action(self, action, raw_obs, topleft=None, botright=None):
        '''
        Computes the raw action corresponding to the chosen abstract action
        :param action: The chosen abstract action (NO_OP, SELECT, RETREAT, or ATTACK)
        :param custom_obs: Custom observations as given by modified state space
        :param raw_obs: Observations from pysc2
        :returns: The raw action to return to environment
        '''
        selected = raw_obs.observation.feature_screen.selected

        if action == Action.SELECT.value:
            assert topleft is None and botright is None, 'Coordinates no longer accepted for select'

            _PLAYER_FRIENDLY = 1
            player_relative = np.array(
                raw_obs.observation.feature_screen.player_relative)
            player_friendly = (player_relative == _PLAYER_FRIENDLY).astype(int)
            num_units = len(raw_obs.observation.feature_units)
            return self._compute_select(player_friendly, num_units)

        if action == Action.ATTACK.value:
            assert not np.all(
                selected == 0), 'Tried to attack when no units selected'
            return actions.FUNCTIONS.Attack_screen('now', topleft)

        if action == Action.MOVE.value:
            assert not np.all(
                selected == 0), 'Tried to move when no units selected'
            return actions.FUNCTIONS.Move_screen('now', topleft)

        if action == Action.STOP.value:
            assert not np.all(
                selected == 0), 'Tried to stop when no units selected'
            return actions.FUNCTIONS.Stop_quick('now')

        if action == Action.NO_OP.value:
            return actions.FUNCTIONS.no_op()

        return actions.FUNCTIONS.no_op()

    def _compute_select(self, friendly_locations, num_units):
        possible_points = np.transpose(np.nonzero(friendly_locations))
        if len(possible_points) == 0 or num_units == 0:
            raise Exception('Actuator cannot select when no units exist')
        if self._select_index >= num_units:
            self._select_index = 0
        idx = int((self._select_index / num_units) * len(possible_points))
        selection = np.flip(possible_points[idx], 0)
        self._select_index += 1.5
        return actions.FUNCTIONS.select_point('select', selection)

    # @staticmethod
    # def _screen_normalize(coords):
    #     coords = coords.reshape((2,))
    #     coords = np.clip(coords, 0, Actuator._SCREEN - 1)
    #     return coords

    # @staticmethod
    # def _select(topleft, botright):

    #     tl = np.array(topleft) * (Actuator._SCREEN /
    #                               (Actuator._SELECT_SPACE-1))
    #     br = np.array(botright) * (Actuator._SCREEN /
    #                                (Actuator._SELECT_SPACE-1))

    #     tl_transform = Actuator._screen_normalize(tl)
    #     br_transform = Actuator._screen_normalize(br)

    #     return actions.FUNCTIONS.select_rect('select', tl_transform, br_transform)
