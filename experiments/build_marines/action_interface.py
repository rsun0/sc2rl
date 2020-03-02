from pysc2.lib import actions, units
from enum import Enum
import numpy as np

NUM_ACTIONS = 6


class BuildMarinesAction(Enum):
    RALLY_SCVS = -1
    NO_OP = 0
    MAKE_SCV = 1
    MAKE_MARINE = 2
    BUILD_DEPOT = 3
    BUILD_BARRACKS = 4
    KILL_MARINE = 5


class BuildMarinesActuator:
    SCV_COST = 50
    MARINE_COST = 50
    DEPOT_COST = 100
    BARRACKS_COST = 100
    DEPOT_LOCATIONS = [(20 + 7*i, 5 + 7*j) for j in range(2) for i in range(9)]
    BARRACKS_LOCATIONS = ([(20 + 10*i, 60) for i in range(6)] + [(83, 60)]
        + [(52 + 10*i, 40) for i in range(4)])
    MAX_SCVS = 22
    MAX_DEPOTS = 18
    MAX_BARRACKS = 11

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.reset()

    def reset(self):
        self.in_progress = None
        self.progress_stage = 0
        self.selecting = False
        self.num_scvs = 12
        self.num_depots = 0
        self.num_barracks = 0

    def compute_action(self, action, obs):
        '''
        Computes the raw action corresponding to the chosen abstract action
        :param action: The chosen abstract BuildMarinesAction
        :param obs: Observations from pysc2
        :returns: The raw action to return to environment
        '''
        assert self.in_progress is None or action is None or action == self.in_progress, \
            'self.in_progress ' + str(self.in_progress) + ', action ' + str(action)

        if self.selecting:
            # Continue SCV selection
            return self._select_scv(obs)

        if action == BuildMarinesAction.RALLY_SCVS:
            return self._rally_scvs(obs)
        if action == BuildMarinesAction.NO_OP:
            return actions.FUNCTIONS.no_op()
        if action == BuildMarinesAction.MAKE_SCV:
            return self._make_scv(obs)
        if action == BuildMarinesAction.MAKE_MARINE:
            return self._make_marine(obs)
        if action == BuildMarinesAction.BUILD_DEPOT:
            return self._build_depot(obs)
        if action == BuildMarinesAction.BUILD_BARRACKS:
            return self._build_barracks(obs)
        if action == BuildMarinesAction.KILL_MARINE:
            return self._kill_marine(obs)

        return actions.FUNCTIONS.no_op()

    def _conclude_sequence(self, obs):
        '''
        Finish an action sequence by resetting Actuator state
        and defaulting to CC selected
        '''
        self.in_progress = None
        self.progress_stage = 0
        return self._select_cc(obs)

    def _rally_scvs(self, obs):
        '''
        Send all idle SCVs to mine
        '''
        self.in_progress = BuildMarinesAction.RALLY_SCVS

        single = obs.observation.single_select
        multi = obs.observation.multi_select
        single_selected = single[0].unit_type == units.Terran.SCV.value
        multi_selected = len(multi) > 0 and all(s.unit_type == units.Terran.SCV.value for s in multi)
        if not (single_selected or multi_selected):
            return actions.FUNCTIONS.select_idle_worker('select_all')

        if self.progress_stage == 0:
            self.progress_stage += 1
            return self._queue_gather(obs)

        return self._conclude_sequence(obs)

    def _select_scv(self, obs):
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

    def _make_scv(self, obs):
        if obs.observation.single_select[0].unit_type != units.Terran.CommandCenter.value:
            self.in_progress = BuildMarinesAction.MAKE_SCV
            return self._select_cc(obs)

        self.in_progress = None
        if self.num_scvs >= self.MAX_SCVS:
            self._print_warning('Reached maximum number of SCVs')
            return actions.FUNCTIONS.no_op()
        if obs.observation.player.minerals < self.SCV_COST:
            self._print_warning('Not enough minerals for SCV')
            return actions.FUNCTIONS.no_op()
        available = obs.observation.available_actions
        if actions.FUNCTIONS.Train_SCV_quick.id not in available:
            self._print_warning('Train_SCV_quick not in available_actions')
            return actions.FUNCTIONS.no_op()
        self.num_scvs += 1
        return actions.FUNCTIONS.Train_SCV_quick('now')
    
    def _make_marine(self, obs):
        self.in_progress = BuildMarinesAction.MAKE_MARINE

        single = obs.observation.single_select
        multi = obs.observation.multi_select
        single_selected = single[0].unit_type == units.Terran.Barracks.value
        multi_selected = len(multi) > 0 and all(s.unit_type == units.Terran.Barracks.value for s in multi)
        if not (single_selected or multi_selected):
            if self.progress_stage > 0:
                self._print_warning('No Barracks to select')
                return self._conclude_sequence(obs)
            self.progress_stage += 1
            return self._select_barracks(obs)

        self.in_progress = None
        self.progress_stage = 0

        if obs.observation.player.minerals < self.MARINE_COST:
            self._print_warning('Not enough minerals for Marine')
            return actions.FUNCTIONS.no_op()
        available = obs.observation.available_actions
        if actions.FUNCTIONS.Train_Marine_quick.id not in available:
            self._print_warning('Train_Marine_quick not in available_actions')
            return actions.FUNCTIONS.no_op()
        return actions.FUNCTIONS.Train_Marine_quick('now')

    def _build_depot(self, obs):
        self.in_progress = BuildMarinesAction.BUILD_DEPOT
        
        if self.progress_stage == 0:
            if self.num_depots >= self.MAX_DEPOTS:
                self._print_warning('Reached maximum number of Supply Depots')
                return self._conclude_sequence(obs)
            if obs.observation.player.minerals < self.DEPOT_COST:
                self._print_warning('Not enough minerals for Supply Depot')
                return self._conclude_sequence(obs)
            self.progress_stage += 1
            return self._select_scv(obs)

        if self.progress_stage == 1:
            available = obs.observation.available_actions
            if actions.FUNCTIONS.Build_SupplyDepot_screen.id not in available:
                self._print_warning('Build_SupplyDepot_screen not in available_actions')
                return self._conclude_sequence(obs)
            self.progress_stage += 1
            return self._place_depot(obs, self.num_depots)

        if self.progress_stage == 2:
            self.progress_stage += 1
            return self._queue_gather(obs)

        assert self.progress_stage == 3 
        self.num_depots += 1
        return self._conclude_sequence(obs)

    def _build_barracks(self, obs):
        MIN_SUPPLY = 23

        self.in_progress = BuildMarinesAction.BUILD_BARRACKS
        
        if self.progress_stage == 0:
            if obs.observation.player.food_cap < MIN_SUPPLY:
                self._print_warning('Cannot build Barracks without Supply Depot')
                return self._conclude_sequence(obs)
            if self.num_barracks >= self.MAX_BARRACKS:
                self._print_warning('Reached maximum number of Barracks')
                return self._conclude_sequence(obs)
            if obs.observation.player.minerals < self.BARRACKS_COST:
                self._print_warning('Not enough minerals for Barracks')
                return self._conclude_sequence(obs)
            self.progress_stage += 1
            return self._select_scv(obs)

        if self.progress_stage == 1:
            available = obs.observation.available_actions
            if actions.FUNCTIONS.Build_Barracks_screen.id not in available:
                self._print_warning('Build_Barracks_screen not in available_actions')
                return self._conclude_sequence(obs)
            self.progress_stage += 1
            return self._place_barracks(obs, self.num_barracks)

        if self.progress_stage == 2:
            self.progress_stage += 1
            return self._queue_gather(obs)

        assert self.progress_stage == 3 
        self.num_barracks += 1
        return self._conclude_sequence(obs)
    
    def _kill_marine(self, obs):
        self.in_progress = BuildMarinesAction.KILL_MARINE

        multi = obs.observation.multi_select
        if len(multi) == 0 or any(s.unit_type != units.Terran.Marine for s in multi):
            available = obs.observation.available_actions
            if self.progress_stage > 0 or actions.FUNCTIONS.select_army.id not in available:
                self._print_warning('Not enough Marines to select')
                return self._conclude_sequence(obs)
            self.progress_stage += 1
            return actions.FUNCTIONS.select_army('select')

        self.in_progress = None
        self.progress_stage = 0
        return self._attack_marine(obs)

    def _print_warning(self, *msg):
        if self.verbose:
            print(*msg)

    @staticmethod
    def _select_cc(obs):
        '''
        Select the command center
        '''
        CC_LOCATION = (35, 29)
        return actions.FUNCTIONS.select_point('select', CC_LOCATION)

    @staticmethod
    def _select_barracks(obs):
        '''
        Select all barracks
        '''
        BARRACKS_LOCATION = BuildMarinesActuator.BARRACKS_LOCATIONS[0]
        return actions.FUNCTIONS.select_point('select_all_type', BARRACKS_LOCATION)

    @staticmethod
    def _queue_gather(obs):
        '''
        Assuming an SCV is selected, queues it back to collecting minerals
        '''
        MINERAL_LOCATION = (14, 14)
        return actions.FUNCTIONS.Harvest_Gather_screen('queued', MINERAL_LOCATION)

        # CENTER_OFFSET = 2
        
        # unit_types = obs.observation.feature_screen.unit_type
        # mineral_locations = np.transpose(np.nonzero(unit_types == units.Neutral.MineralField.value))
        # mineral_location = np.flip(mineral_locations[0], 0)
        # # Select center of mineral instead of corner
        # mineral_location += CENTER_OFFSET
        # return actions.FUNCTIONS.Harvest_Gather_screen('queued', mineral_location)

    @staticmethod
    def _place_depot(obs, location_index):
        '''
        Assuming an SCV is selected, builds a supply depot
        '''
        build_location = BuildMarinesActuator.DEPOT_LOCATIONS[location_index]
        return actions.FUNCTIONS.Build_SupplyDepot_screen('now', build_location)
    
    @staticmethod
    def _place_barracks(obs, location_index):
        '''
        Assuming an SCV is selected, builds a barracks
        '''
        build_location = BuildMarinesActuator.BARRACKS_LOCATIONS[location_index]
        return actions.FUNCTIONS.Build_Barracks_screen('now', build_location)

    # FIXME sometimes shoots barracks
    @staticmethod
    def _attack_marine(obs):
        '''
        Queues an attack on a random Marine
        '''
        MIN_DENSITY = 16

        marine_locations = obs.observation.feature_screen.unit_type == units.Terran.Marine.value
        dense_locations = obs.observation.feature_screen.unit_density_aa >= MIN_DENSITY

        marine_locations = np.logical_and(marine_locations, dense_locations)
        marine_locations = np.flip(np.transpose(np.nonzero(marine_locations)), axis=1)
        rand_index = np.random.randint(marine_locations.shape[0])
        attack_location = marine_locations[rand_index]
        return actions.FUNCTIONS.Attack_screen('queued', attack_location)

    def __str__(self):
        return ('SCVs: ' + str(self.num_scvs) + ', Depots: ' + str(self.num_depots)
            + ', Rax: ' + str(self.num_barracks))