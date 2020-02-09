import sys
sys.path.insert(0, "../interface/")

from agent import Agent
from action_interface import BuildMarinesAction

class TestAgent(Agent):
    def __init__(self):
        # Intentionally bypassing parent constructor
        self.built_depot = False

    def _sample(self, state):
        if state.observation.player.minerals < 100:
            return BuildMarinesAction.NO_OP
        if not self.built_depot:
            self.built_depot = True
            return BuildMarinesAction.BUILD_DEPOT
        return BuildMarinesAction.BUILD_BARRACKS

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