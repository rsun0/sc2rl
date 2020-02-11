import sys
sys.path.insert(0, "../interface/")

from agent import Agent
from action_interface import BuildMarinesAction

class TestAgent(Agent):
    def __init__(self):
        # Intentionally bypassing parent constructor
        self.num_depots = 0
        self.num_barracks = 0
        self.num_scvs = 12
        self.time = 0
        self.wait = 0

    def _sample(self, state):
        if self.wait > 0:
            self.wait -= 1
            return BuildMarinesAction.NO_OP
        self.time += 1
        if state.observation.player.minerals < 300:
            return BuildMarinesAction.NO_OP
        if self.num_depots < 1:
            self.num_depots += 1
            self.wait = 61
            return BuildMarinesAction.BUILD_DEPOT
        if self.num_barracks < 1:
            self.num_barracks += 1
            return BuildMarinesAction.BUILD_BARRACKS
        return BuildMarinesAction.MAKE_MARINE

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