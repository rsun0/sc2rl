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
    BARRACKS_COST = 100
    DEPOT_LOCATIONS = [(20 + 7*i, 5 + 7*j) for j in range(2) for i in range(9)]
    BARRACKS_LOCATIONS = ([(20 + 10*i, 60) for i in range(6)] + [(83, 60)]
        + [(52 + 10*i, 40) for i in range(4)])
    MAX_DEPOTS = 2
    MAX_BARRACKS = 11

    def __init__(self):
        self.reset()

    def reset(self):
        self.in_progress = None
        self.progress_stage = 0
        self.selecting = False
        self.num_depots = 0
        self.num_barracks = 0

    def compute_action(self, action, raw_obs):
        '''
        Computes the raw action corresponding to the chosen abstract action
        :param action: The chosen abstract BuildMarinesAction
        :param raw_obs: Observations from pysc2
        :returns: The raw action to return to environment
        '''
        assert self.in_progress is None or action is None or action == self.in_progress

        if self.selecting:
            # Continue SCV selection
            return self._select_scv(raw_obs)

        if action == BuildMarinesAction.MAKE_SCV:
            # TODO 22 SCVs is optimal, 23rd is useless
            pass
        if action == BuildMarinesAction.NO_OP:
            return actions.FUNCTIONS.no_op()
        if action == BuildMarinesAction.BUILD_DEPOT:
            return self._build_depot(raw_obs)
        if action == BuildMarinesAction.BUILD_BARRACKS:
            return self._build_barracks(raw_obs)
        if action == BuildMarinesAction.KILL_MARINE:
            # TODO KILL_MARINE should be queued action, not now
            pass

        return actions.FUNCTIONS.no_op()

    def _conclude_sequence(self, raw_obs):
        '''
        Finish an action sequence by resetting Actuator state
        and defaulting to CC selected
        '''
        self.in_progress = None
        self.progress_stage = 0
        return self._select_cc(raw_obs)

    def _select_scv(self, raw_obs):
        '''
        Select the leftmost SCV (SCVs on the left of the map are mining)
        '''
        MINING_TOPLEFT = (10, 19)
        MINING_BOTRIGHT = (26, 48)
        
        if not self.selecting:
            self.selecting = True
            return actions.FUNCTIONS.select_rect('select', MINING_TOPLEFT, MINING_BOTRIGHT)
        else:
            self.selecting = False
            return actions.FUNCTIONS.select_unit('select', 0)

    def _build_depot(self, raw_obs):
        self.in_progress = BuildMarinesAction.BUILD_DEPOT
        
        if self.progress_stage == 0:
            if self.num_depots >= self.MAX_DEPOTS:
                print('Reached maximum number of Supply Depots')
                return self._conclude_sequence(raw_obs)
            if raw_obs.observation.player.minerals < self.DEPOT_COST:
                print('Not enough minerals for Supply Depot')
                return self._conclude_sequence(raw_obs)
            self.progress_stage += 1
            return self._select_scv(raw_obs)

        if self.progress_stage == 1:
            available = raw_obs.observation.available_actions
            if actions.FUNCTIONS.Build_SupplyDepot_screen.id not in available:
                print('Build_SupplyDepot_screen not in available_actions')
                return self._conclude_sequence(raw_obs)
            self.progress_stage += 1
            return self._place_depot(raw_obs, self.num_depots)

        if self.progress_stage == 2:
            self.progress_stage += 1
            return self._queue_gather(raw_obs)

        assert self.progress_stage == 3 
        self.num_depots += 1
        return self._conclude_sequence(raw_obs)

    def _build_barracks(self, raw_obs):
        MIN_SUPPLY = 23

        self.in_progress = BuildMarinesAction.BUILD_BARRACKS
        
        if self.progress_stage == 0:
            if raw_obs.observation.player.food_cap < MIN_SUPPLY:
                print('Cannot build Barracks without Supply Depot')
                return self._conclude_sequence(raw_obs)
            if self.num_barracks >= self.MAX_BARRACKS:
                print('Reached maximum number of Barracks')
                return self._conclude_sequence(raw_obs)
            if raw_obs.observation.player.minerals < self.BARRACKS_COST:
                print('Not enough minerals for Barracks')
                return self._conclude_sequence(raw_obs)
            self.progress_stage += 1
            return self._select_scv(raw_obs)

        if self.progress_stage == 1:
            available = raw_obs.observation.available_actions
            if actions.FUNCTIONS.Build_Barracks_screen.id not in available:
                print('Build_Barracks_screen not in available_actions')
                return self._conclude_sequence(raw_obs)
            self.progress_stage += 1
            return self._place_barracks(raw_obs, self.num_barracks)

        if self.progress_stage == 2:
            self.progress_stage += 1
            return self._queue_gather(raw_obs)

        assert self.progress_stage == 3 
        self.num_barracks += 1
        return self._conclude_sequence(raw_obs)

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
    def _queue_gather(raw_obs):
        '''
        Assuming an SCV is selected, queues it back to collecting minerals
        '''
        MINERAL_LOCATION = (14, 14)
        return actions.FUNCTIONS.Harvest_Gather_screen('queued', MINERAL_LOCATION)

    @staticmethod
    def _place_depot(raw_obs, location_index):
        '''
        Assuming an SCV is selected, builds a supply depot
        '''
        build_location = BuildMarinesActuator.DEPOT_LOCATIONS[location_index]
        return actions.FUNCTIONS.Build_SupplyDepot_screen('now', build_location)
    
    @staticmethod
    def _place_barracks(raw_obs, location_index):
        '''
        Assuming an SCV is selected, builds a barracks
        '''
        build_location = BuildMarinesActuator.BARRACKS_LOCATIONS[location_index]
        return actions.FUNCTIONS.Build_Barracks_screen('now', build_location)