from pysc2.lib import actions, units
from enum import Enum
import numpy as np


class BuildMarinesAction(Enum):
    NO_OP = 0
    MAKE_SCV = 1
    MAKE_MARINE = 2
    BUILD_DEPOT = 3
    BUILD_BARRACKS = 4
    KILL_MARINE = 5


class BuildMarinesActuator:

    def __init__(self):
        self.reset()

    def reset(self):
        pass

    def compute_action(self, action, raw_obs):
        '''
        Computes the raw action corresponding to the chosen abstract action
        :param action: The chosen abstract BuildMarinesAction
        :param raw_obs: Observations from pysc2
        :returns: The raw action to return to environment
        '''
        selected = raw_obs.observation.feature_screen.selected

        if action == BuildMarinesAction.BUILD_DEPOT:
            # FIXME
            return self._select_scv(raw_obs)

        if action == BuildMarinesAction.NO_OP:
            return actions.FUNCTIONS.no_op()

        return actions.FUNCTIONS.no_op()

    @staticmethod
    def _select_scv(raw_obs):
        '''
        Select the leftmost SCV (SCVs on the left of the map are mining)
        '''
        unit_types = raw_obs.observation.feature_screen.unit_type
        scv_locations = np.transpose(np.nonzero(unit_types == units.Terran.SCV.value))
        # TODO which way is left? NOT THIS WAY
        # TODO currently selecting top
        select_point = np.flip(scv_locations[0], 0)
        return actions.FUNCTIONS.select_point('select', select_point)

    @staticmethod
    def _build_depot(raw_obs):
        '''
        Assuming an SCV is selected, builds a supply depot
        '''
        # TODO compute free location to build
        build_location = 0
        return actions.FUNCTIONS.Build_SupplyDepot_screen('now', build_location)