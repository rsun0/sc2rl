import sys
sys.path.insert(0, "../interface/")

from pysc2.lib import units
from agent import Agent
from action_interface import BuildMarinesAction

class TestAgent(Agent):
    NOT_SET = 999999

    def __init__(self):
        # Intentionally bypassing parent constructor
        self.reset()

    def reset(self):
        self.num_depots = 0
        self.num_barracks = 0
        self.num_scvs = 12
        self.rax_done_at = self.NOT_SET

    def push_memory_terminal(self, _):
        self.reset()

    def _sample(self, state):
        mins = state.observation.player.minerals
        time = state.observation.game_loop[0]

        if mins < 50:
            # if state.observation.player.food_army > 8:
            #     return BuildMarinesAction.KILL_MARINE
            return BuildMarinesAction.NO_OP

        # if self.num_scvs < 22:
        #     if (state.observation.single_select[0].unit_type == units.Terran.CommandCenter.value
        #         and len(state.observation.build_queue) == 0):
        #         self.num_scvs += 1
        #         return BuildMarinesAction.MAKE_SCV

        if self.num_depots == 0 and mins >= 100:
            self.num_depots += 1
            return BuildMarinesAction.BUILD_DEPOT

        if self.num_barracks < 7 and state.observation.player.food_cap >= 23:
            if mins >= 150:
                self.num_barracks += 1
                if self.rax_done_at == self.NOT_SET:
                    self.rax_done_at = time + 1110
                return BuildMarinesAction.BUILD_BARRACKS
            else:
                return BuildMarinesAction.NO_OP
        
        if time >= self.rax_done_at and state.observation.player.food_used < state.observation.player.food_cap:
            return BuildMarinesAction.MAKE_MARINE
        
        if self.num_depots < 3 and mins >= 100:
            self.num_depots += 1
            return BuildMarinesAction.BUILD_DEPOT
        
        # if state.observation.player.food_army > 8:
        #     return BuildMarinesAction.KILL_MARINE
        return BuildMarinesAction.NO_OP

    def _forward(self, state):
        return self._sample(state)

    def state_space_converter(self, state):
        return state

    def action_space_converter(self, action):
        return action

    def train(self, run_settings):
        pass

    def train_step(self, batch_size):
        pass

    def save(self):
        pass
    
    def push_memory(self, state, action, reward, done):
        pass