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
        self.in_progress = None
        self.depot_placed = False
        self.barracks_placed = False

    def compute_action(self, action, raw_obs):
        '''
        Computes the raw action corresponding to the chosen abstract action
        :param action: The chosen abstract BuildMarinesAction
        :param raw_obs: Observations from pysc2
        :returns: The raw action to return to environment
        '''
        assert self.in_progress is None or action is None or action == self.in_progress

        if action == BuildMarinesAction.NO_OP:
            return actions.FUNCTIONS.no_op()
        
        if action == BuildMarinesAction.BUILD_DEPOT:
            return self._build_depot(raw_obs)

        return actions.FUNCTIONS.no_op()

    def _build_depot(self, raw_obs):
        self.in_progress = BuildMarinesAction.BUILD_DEPOT

        selected = raw_obs.observation.single_select
        # NOTE Assumes selected has 1 element
        if selected[0].unit_type != units.Terran.SCV.value:
            # FIXME Cannot succesfully select SCVs
            print('SELECTING')
            return self._select_scv(raw_obs)
        
        if not self.depot_placed:
            print('PLACING')
            self.depot_placed = True
            return self._place_depot(raw_obs)
        
        self.in_progress = None
        self.depot_placed = False
        return self._queue_gather(raw_obs)

    @staticmethod
    def _select_scv(raw_obs):
        '''
        Select the leftmost SCV (SCVs on the left of the map are mining)
        '''
        unit_types = raw_obs.observation.feature_screen.unit_type
        scv_locations = np.transpose(np.nonzero(unit_types == units.Terran.SCV.value))
        # Switch x-coordinate to be 1st
        scv_locations = np.flip(scv_locations, 1)
        select_point = np.amin(scv_locations, axis=0)
        return actions.FUNCTIONS.select_point('select', select_point)

    @staticmethod
    def _place_depot(raw_obs):
        '''
        Assuming an SCV is selected, builds a supply depot
        '''
        # Check if enough minerals
        DEPOT_COST = 100
        minerals = raw_obs.observation.player.minerals
        if minerals < DEPOT_COST:
            return actions.FUNCTIONS.no_op()
        # TODO compute free location to build
        build_location = (0, 5)
        # TODO decide correct screen dimensions by observing Depot height and width, possibly 640 x 480
        print(raw_obs.observation.feature_screen.unit_density_aa)
        return actions.FUNCTIONS.Build_SupplyDepot_screen('now', build_location)

    @staticmethod
    def _queue_gather(raw_obs):
        '''
        Assuming an SCV is selected, queues it back to collecting minerals
        '''
        unit_types = raw_obs.observation.feature_screen.unit_type
        mineral_locations = np.transpose(np.nonzero(unit_types == units.Neutral.MineralField.value))
        mineral_location = np.flip(mineral_locations[0], 0)
        print(mineral_locations)
        print(mineral_location)
        return actions.FUNCTIONS.Harvest_Gather_screen('queued', mineral_location)