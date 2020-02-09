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
    DEPOT_COST = 100
    DEPOT_LOCATIONS = [(30 + 7*i, 5 + 7*j) for j in range(3) for i in range(8)]
    MAX_DEPOTS = 23

    def __init__(self):
        self.reset()

    def reset(self):
        self.in_progress = None
        self.progress_stage = 0
        self.num_depots = 0

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

        if self.num_depots >= self.MAX_DEPOTS:
            print('Reached maximum number of Supply Depots')
            return self._conclude_sequence(raw_obs)

        # NOTE Assumes selected has 1 element
        selected = raw_obs.observation.single_select
        if selected[0].unit_type != units.Terran.SCV.value:
            return self._select_scv(raw_obs)
        elif self.progress_stage == 0:
            self.progress_stage += 1

        if self.progress_stage == 1:
            available = raw_obs.observation.available_actions
            if actions.FUNCTIONS.Build_SupplyDepot_screen.id not in available:
                print('Cannot build depot')
                return self._conclude_sequence(raw_obs)
            
            self.progress_stage += 1
            return self._place_depot(raw_obs, self.num_depots)

        if self.progress_stage == 2:
            self.progress_stage += 1
            return self._queue_gather(raw_obs)

        assert self.progress_stage == 3 
        self.num_depots += 1
        return self._conclude_sequence(raw_obs)

    def _conclude_sequence(self, raw_obs):
        '''
        Finish an action sequence by resetting Actuator state
        and defaulting to CC selected
        '''
        self.in_progress = None
        self.progress_stage = 0
        return self._select_cc(raw_obs)

    # FIXME Sometimes misses an SCV
    @staticmethod
    def _select_scv(raw_obs):
        '''
        Select the leftmost SCV (SCVs on the left of the map are mining)
        '''
        TOPLEFT_OFFSET = 2
        BOTRIGHT_OFFSET = 4
        
        unit_types = raw_obs.observation.feature_screen.unit_type
        scv_locations = np.transpose(np.nonzero(unit_types == units.Terran.SCV.value))
        # Switch x-coordinate to be 1st
        scv_locations = np.flip(scv_locations, 1)
        select_point = np.amin(scv_locations, axis=0)
        # Select center of SCV instead of corner
        select_topleft = select_point - TOPLEFT_OFFSET
        select_botright = select_point + BOTRIGHT_OFFSET
        return actions.FUNCTIONS.select_rect('select', select_topleft, select_botright)

    @staticmethod
    def _select_cc(raw_obs):
        '''
        Select the command center
        '''
        CENTER_OFFSET = 5
        
        unit_types = raw_obs.observation.feature_screen.unit_type
        cc_locations = np.transpose(np.nonzero(unit_types == units.Terran.CommandCenter.value))
        # Switch x-coordinate to be 1st
        cc_location = np.flip(cc_locations[0], axis=0)
        # Select center of CC instead of corner
        cc_location += CENTER_OFFSET
        return actions.FUNCTIONS.select_point('select', cc_location)

    @staticmethod
    def _place_depot(raw_obs, location_index):
        '''
        Assuming an SCV is selected, builds a supply depot
        '''
        build_location = BuildMarinesActuator.DEPOT_LOCATIONS[location_index]
        return actions.FUNCTIONS.Build_SupplyDepot_screen('now', build_location)

    @staticmethod
    def _queue_gather(raw_obs):
        '''
        Assuming an SCV is selected, queues it back to collecting minerals
        '''
        CENTER_OFFSET = 2
        
        unit_types = raw_obs.observation.feature_screen.unit_type
        mineral_locations = np.transpose(np.nonzero(unit_types == units.Neutral.MineralField.value))
        mineral_location = np.flip(mineral_locations[0], 0)
        # Select center of mineral instead of corner
        mineral_location += CENTER_OFFSET
        return actions.FUNCTIONS.Harvest_Gather_screen('queued', mineral_location)